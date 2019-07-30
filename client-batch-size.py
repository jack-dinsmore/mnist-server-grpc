import logging, grpc, time

import numpy as np
import matplotlib.pyplot as plt

import server_tools_pb2
import server_tools_pb2_grpc

PORT = '50051'
f = open("IP.txt")
IP = f.read()
if IP[-1] == '\n':
    IP = IP[:-1]
f.close()

NUM_IMAGES=200
NUM_TRIALS=100

def run():
    # Get a handle to the server
    channel = grpc.insecure_channel(IP+':'+PORT)
    stub = server_tools_pb2_grpc.MnistServerStub(channel)

    # Get a client ID which you need to talk to the server
    try:
        response = stub.RequestClientID(server_tools_pb2.NullParam())
    except:
        print("Connection to the server could not be established. Press enter to try again.")
        return
    client_id = response.new_id

    batch_sizes = list(range(1, 10)) + list(range(10, 100, 10))
    ideal_time = []
    real_time = []

    for batch_size in batch_sizes:
        print(batch_size)
        real_time_now = 0
        ideal_time_now = 0
        for i in range(NUM_TRIALS):
            data = np.random.rand(NUM_IMAGES, 28, 28, 1).tostring()
            start_time=time.time()
            response = stub.StartJobWait(server_tools_pb2.DataMessage(images=data, batch_size=batch_size, client_id = client_id))
            original_array = np.frombuffer(response.prediction).reshape(NUM_IMAGES, 10)
            real_time_now += time.time() - start_time
            ideal_time_now += response.infer_time
        ideal_time.append(ideal_time_now / NUM_IMAGES / NUM_TRIALS*1000)
        real_time.append(real_time_now / NUM_IMAGES / NUM_TRIALS*1000)

    i = input()
    while i != '':
        plt.plot(batch_sizes, real_time, 'r')#c='b', marker='o', alpha=0.5)
        plt.plot(batch_sizes, ideal_time, 'b')#c='r', marker='s', alpha=0.5)

        plt.legend(['Actual', 'Ideal'])
        plt.title('Inference time vs batch size, 200 images')
        plt.xlabel('Batch size')
        plt.xscale('log')
        plt.axis([1, 100, 0, float(i)])
        plt.ylabel('Inference time per image (ms)')
        plt.show()
        i = input()


    channel.close()


if __name__ == '__main__':
    logging.basicConfig()
    # Repeat so that you can change the image
    run()
