import logging, grpc, time
import numpy as np

import server_tools_pb2
import server_tools_pb2_grpc

PORT = '50051'
f = open("IP.txt")
IP = f.read()
if IP[-1] == '\n':
    IP = IP[:-1]
f.close()

# Set this flag to indicate whether the client should wait until the prediction
# is finished or check in with the server periodically until it is 
WAIT = False

# Change this parameter depending on how many images you want to send at once.
# There is an upper limit (668 on my machine) where the size of the package 
# becomes too great and the client will throw an error.
NUM_IMAGES = 668

def run():
    # Get a handle to the server
    channel = grpc.insecure_channel(IP + ':' + PORT)
    stub = server_tools_pb2_grpc.MnistServerStub(channel)

    # Get a client ID which you need to talk to the server
    try:
        response = stub.RequestClientID(server_tools_pb2.NullParam())
    except:
        print("Connection to the server could not be established. Press enter to try again.")
        return
    client_id = response.new_id

    # Generate lots of data
    data = np.random.rand(NUM_IMAGES, 28, 28, 1)
    data = data.tostring()

    # Send the data to the server and receive an answer
    start_time = time.time()
    if WAIT:
        print("Submitting images and waiting")
        response = stub.StartJobWait(server_tools_pb2.DataMessage(images=data, num_images=NUM_IMAGES, client_id=client_id))
    else:
        print("Submitting images")
        try:
            idPackage = stub.StartJobNoWait(server_tools_pb2.DataMessage(images=data, num_images=NUM_IMAGES, client_id=client_id))
        except:
            print("NUM_IMAGES is too high")
            return
        response = stub.ProbeJob(idPackage)
        print("Checking in with server")
        while not response.complete:
            response = stub.ProbeJob(idPackage)
            if response.error != '':
                print(response.error)
                break

    # Print output
    original_array = np.frombuffer(response.prediction).reshape(NUM_IMAGES, 10)
    whole_time = time.time() - start_time
    print("Total time:", whole_time)
    print("Predict time:", response.infer_time)
    print("Fraction of time spent not predicting:", (1 - response.infer_time / whole_time) * 100, '%')
    channel.close()


if __name__ == '__main__':
    logging.basicConfig()
    while input() == '':
        run()
