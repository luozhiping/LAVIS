import cv2, os, time

for root, dirs, files in os.walk("./", topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        # print(filename)
        if 'KZSQ4575.MP4' in filename:
            print("*********************")
            print(filename, name)
            cap = cv2.VideoCapture(filename)
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
#                cv2.imshow('capture', frame)
                cv2.imwrite("./images/%s__%s.jpg" % (name, i), frame)
                i = i + 1
                print(i)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
    # for name in dirs:
