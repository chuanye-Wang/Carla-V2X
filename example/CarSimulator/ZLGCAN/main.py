from ControlCAN import *
from Storage import *
import msvcrt
import configparser


def CAN_Config():
    cf = configparser.ConfigParser()
    cf.read('config.ini')
    can_devtype = cf.getint("can", "devicetype")
    can_devindex = cf.getint("can", "deviceindex")
    can_canindex = cf.getint("can", "canindex")
    can_baudrate = cf.getint("can", "baudrate")
    can_acccode = int(cf.get("can", "acceptcode"), 16)
    can_accmask = int(cf.get("can", "acceptmask"), 16)
    # db_ip = cf.get("db", "ip")
    # db_user = cf.get("db", "username")
    # db_pass = cf.get("db", "password")
    # db_schema = cf.get("db", "schema")
    # db_rtable = cf.get("db", "rawtable")
    # db_ttable = cf.get("db", "turetable")
    # db_buffersize = cf.getint("db", "buffersize")
    print('读取配置成功')

    # sql = StorageToSQL(db_ip, db_user, db_pass, db_schema, db_rtable, db_ttable, db_buffersize)
    # sql.createtable()
    can = ControlCAN(can_devtype, can_devindex, can_canindex, can_baudrate, can_acccode, can_accmask)
    can.opendevice()
    can.initcan()
    can.startcan()
    while 1:
        if kbq(): break
        can.receive()
        # sql.copy(can.receivebuf, can.receivenum, can.timeinterval)
        # sql.storage()
        # sql.commit()
    del can
    # del sql


def kbq():
    if msvcrt.kbhit():
        ret = ord(msvcrt.getch())
        if ret == 113 or ret == 81:  # q or Q
            return 1


if __name__ == "__main__":
    CAN_Config()
