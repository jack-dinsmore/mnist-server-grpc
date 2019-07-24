# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent import futures
import logging, grpc, time
import numpy as np
import ml_functions as ml
import multiprocessing as mp

import server_tools_pb2
import server_tools_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
PORT='50051'

global processes, max_id, manager, results, max_client_ids, max_client_id, new_client_permitted, times
processes = {}
max_client_ids = {}
max_client_id = 0
max_id = 0
new_client_permitted = True

# Data accessible to threads and the main program
initialized = False
manager = None
results = None
times = None

def initialize():
    global manager, results, times, initialized
    if not initialized:
        manager = mp.Manager()
        results = manager.dict()
        times = manager.dict()
        initialized = True


class MnistServer(server_tools_pb2_grpc.MnistServerServicer):

    def StartJobWait(self, request, context):
        initialize()
        data = np.frombuffer(request.images)
        data = data.reshape(request.num_images, 28, 28, 1)
        prediction, predict_time = ml.predict(data)
        return server_tools_pb2.PredictionMessage(complete=True, prediction=prediction.tostring(), error='', time=predict_time)

    def RequestClientID(self, request, context):
        initialize()
        global max_client_id, new_client_permitted, max_client_ids
        while not new_client_permitted:
            pass

        new_client_permitted = False
        id = str(max_client_id)
        max_client_id += 1
        new_client_permitted = True

        max_client_ids[id] = 0
        return server_tools_pb2.IDMessage(id=id, error = '')

    def StartJobNoWait(self, request, context):
        global processes, results, max_client_ids
        initialize()
        if request.client_id not in max_client_ids:
            return server_tools_pb2.IDMessage(id=None, error = "The ID "+str(request.client_id)+" is not a valid client ID")
        
        data = np.frombuffer(request.images)
        data = data.reshape(request.num_images, 28, 28, 1)

        id = request.client_id + '-' + str(max_client_ids[request.client_id])
        max_client_ids[request.client_id] += 1

        results[id] = None
        processes[id] = mp.Process(target=ml.predict, args=(data, results, times, id))
        processes[id].start()
        return server_tools_pb2.IDMessage(id=id, error='')
    
    def ProbeJob(self, request, context):
        global processes, results
        initialize()
        if request.id not in processes:
            return server_tools_pb2.PredictionMessage(complete=False, prediction=None, 
                error = "The ID "+str(request.id)+" is not a valid job ID")
        if results[request.id] is None:
            a = results[request.id]
            return server_tools_pb2.PredictionMessage(complete=False, prediction=None)
        else:
            prediction = results[request.id].tostring()
            # Am I supposed to join the process to prevent memory leaks or something?
            del processes[request.id]
            del results[request.id]
            return server_tools_pb2.PredictionMessage(complete=True, prediction=prediction, error='', time=times[request.id])


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_tools_pb2_grpc.add_MnistServerServicer_to_server(MnistServer(), server)
    server.add_insecure_port('[::]:'+PORT)
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
