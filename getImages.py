import urllib.request as urllib
from PIL import Image, ImageTk
import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def getClothing(item):
    with open("./URLS/" + item + ".txt", "r") as lines:
        for i, line in enumerate(lines):
            try:
                with timeout(seconds=2):
                    print("Writing image number " + str(i))
                    #resource = urllib.urlopen(line)
                    urllib.urlretrieve(line, "./Data/" + item + "/" + item + str(i) +".jpg")
                    im1 = Image.open("./Data/" + item + "/" + item + str(i) +".jpg")
                    im_small = im1.resize((256, 256), Image.ANTIALIAS)
                    im_small.save("./Data/" + item + "/" + item + str(i) +".jpg")

            except:
                print("Error: " + str(i))
                continue


getClothing("bag")
getClothing("dress")
