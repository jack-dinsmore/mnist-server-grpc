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
print(IP)

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

    # Load the image from image.bmp
    image = open('image.bmp', 'rb')
    b = image.read()[54:]# Remove header
    image.close()
    assert(len(b)==28*28*3)# Make sure the image has no transparency and is 28x28
    data = np.ndarray((1, 28, 28, 1))
    for y in range(28):
        for x in range(28):
            i = 3*(28*y+x)
            data[0][27-y][x][0] = 1 - (int(b[i])+int(b[i+1])+int(b[i+2])) / 3 / 255
    data = data.tostring()

    # Pass the data to the server and receive a prediction
    print("Submitting image and waiting")
    start_time=time.time()
    response = stub.StartJobWait(server_tools_pb2.DataMessage(images=data, num_images=1, client_id = client_id))

    # Find the most likely prediction and print it
    original_array = np.frombuffer(response.prediction).reshape(1, 10)
    whole_time = time.time() - start_time
    result = list(original_array[0])
    print("Prediction is:", result.index(max(result)))
    print("Total time:", whole_time)
    print("Predict time:", response.infer_time)
    print("Fraction of time spent not predicting:", (1 - response.infer_time / whole_time) * 100, '%')
    channel.close()


if __name__ == '__main__':
    logging.basicConfig()
    # Repeat so that you can change the image
    while input('Change image.bmp if you like') == '':
        run()
