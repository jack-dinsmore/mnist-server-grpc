import logging, grpc, time
import numpy as np

import server_tools_pb2
import server_tools_pb2_grpc

from matplotlib import pyplot as plt

PORT = '50051'
f = open("IP.txt")
IP = f.read()
if IP[-1] == '\n':
    IP = IP[:-1]
f.close()

def run(wait, num_images):
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
    data = np.random.rand(num_images, 28, 28, 1)
    data = data.tostring()

    # Send the data to the server and receive an answer
    start_time = time.time()
    if wait:
        response = stub.StartJobWait(server_tools_pb2.DataMessage(images=data, num_images=num_images, client_id=client_id))
    else:
        idPackage = stub.StartJobNoWait(server_tools_pb2.DataMessage(images=data, num_images=num_images, client_id=client_id))
        response = stub.ProbeJob(idPackage)
        while not response.complete:
            response = stub.ProbeJob(idPackage)
            if response.error != '':
                print(response.error)
                break

    # Print output
    original_array = np.frombuffer(response.prediction).reshape(num_images, 10)
    whole_time = time.time() - start_time
    fraction_not_predicting = (1 - response.infer_time / whole_time)
    channel.close()
    return fraction_not_predicting


if __name__ == '__main__':
    logging.basicConfig()
    wait = True
    image_number = list(range(1, 10))+list(range(10, 100, 10))
    wait_fraction = []
    no_wait_fraction = []
    print("WAITING")
    for num_images in image_number:
        print(num_images)
        wait_fraction.append(run(True, num_images))
    print("NOT WAITING")
    for num_images in image_number:
        print(num_images)
        no_wait_fraction.append(run(False, num_images))

    plt.scatter(image_number, wait_fraction, c='r', marker='o', alpha=0.5)
    plt.scatter(image_number, no_wait_fraction, c='b', marker='s', alpha=0.5)
    plt.legend(['Wait', 'No wait'])
    plt.xlabel('Images sent')
    plt.xscale('log')
    plt.ylabel('Fraction time spent not predicting')
    plt.show()