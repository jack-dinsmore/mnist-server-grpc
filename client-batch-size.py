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
    ideal_time = [0]*len(batch_sizes)
    real_time = [0]*len(batch_sizes)
    print(client_id)

    for i in range(NUM_TRIALS):
        print("\nTrial", i)
        j = 0
        for batch_size in batch_sizes:
            print(j, end=',')
            data = np.random.rand(NUM_IMAGES, 28, 28, 1).tostring()
            start_time=time.time()
            response = stub.StartJobWait(server_tools_pb2.DataMessage(images=data, batch_size=batch_size, client_id=client_id))
            original_array = np.frombuffer(response.prediction).reshape(NUM_IMAGES, 10)
            ideal_time[j] += response.infer_time
            real_time[j] += time.time() - start_time
            j += 1
    
    for i in range(len(batch_sizes)):
        ideal_time[i] /= NUM_IMAGES * NUM_TRIALS / 1000 # Convert to ms
        real_time[i] /= NUM_IMAGES * NUM_TRIALS / 1000 # Convert to ms

    i = input()
    while i != '':
        plt.plot(batch_sizes, real_time, 'r')
        plt.plot(batch_sizes, ideal_time, 'b')

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
