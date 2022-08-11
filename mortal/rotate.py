import prelude

from datetime import datetime
import logging
from time import sleep


def cp(source, target):
    with open(source, "rb") as source_file, open(target, "wb") as tar_file:
        logging.info("{} -> {}".format(source, target))
        tar_file.write(source_file.read())


def rotate():
    curent = "data/mortal.pth"
    snapshot = "data/snap.pth"
    yesterday = "data/yes.pth"
    archive = "data/" + datetime.now().strftime("%Y%m%d") + ".pth"
    cp(snapshot,yesterday)
    cp(curent,snapshot)
    cp(curent,archive)

def main():
    while True:
        rotate()
        sleep(60*60*24)

if __name__ == "__main__":
    main()