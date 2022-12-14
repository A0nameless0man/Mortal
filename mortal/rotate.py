import prelude

from datetime import timedelta, datetime, date, time
import logging
import os
from time import sleep
from config import config

interval_hour = 6

assert 24 % interval_hour == 0

def cp(source, target) -> bool:
    if os.path.exists(source):
        with open(source, "rb") as source_file, open(target, "wb") as tar_file:
            logging.info("{} -> {}".format(source, target))
            tar_file.write(source_file.read())
            return True
    return False


def rotate():
    curent = config["control"]["state_file"]
    archive_dir = config["control"]["archive_dir"]
    archive = os.path.join(
        archive_dir,
        os.path.splitext(os.path.basename(curent))[0]
        + "-"
        + datetime.now().strftime("%Y%m%d%H")
        + ".pth"
    )
    cp(curent, archive)

    # for i in range(6):
    #     yesterday = os.path.dirname(curent) + "/T-{}.pth".format(i)
    #     old_archive = (
    #         os.path.dirname(curent)
    #         + "/"
    #         + os.path.splitext(os.path.basename(curent))[0]
    #         + "-"
    #         + (datetime.now() + timedelta(hours=(-(i*interval_hour)))).strftime("%Y%m%d%H")
    #         + ".pth"
    #     )
    #     if not cp(old_archive, yesterday):
    #         old_archive = os.path.dirname(curent) + "/T-{}.pth".format(i - 1)
    #         cp(old_archive, yesterday)


def sleep_to_dawn():
    now = datetime.today()

    next_dawn_day = date.today()
    dawn_time = time(hour=0, minute=0)
    if now.time() > dawn_time:
        next_dawn_day += timedelta(days=1)
    next_dawn = datetime.combine(next_dawn_day, dawn_time)
    delt = next_dawn - now
    delt = delt.total_seconds()
    logging.info("sleeping {} seconds".format(delt))
    sleep(delt)

def sleep_to_hour():
    now = datetime.today()
    t=now.time()
    h=(t.hour//interval_hour)*interval_hour
    dawn_time = time(hour=(h+interval_hour)%24, minute=0)
    next_dawn_day = date.today()
    next_dawn_day += timedelta(days=(h+interval_hour)//24)
    next_dawn = datetime.combine(next_dawn_day, dawn_time)
    delt = next_dawn - now
    delt = delt.total_seconds()
    logging.info("sleeping {} seconds".format(delt))
    sleep(delt)

def main():
    while True:
        rotate()
        sleep_to_hour()


if __name__ == "__main__":
    main()
