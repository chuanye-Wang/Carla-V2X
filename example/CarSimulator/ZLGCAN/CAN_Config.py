from ControlCAN import *
from Storage import *
import msvcrt
import configparser
import os

class CAN_Config():
    def __init__(self):
        cf = configparser.ConfigParser()
        configPath=os.getcwd()
        filepath=os.path.join(configPath,'ZLGCAN\CAN_config.ini')
        cf.read(filepath)
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
        self.can = ControlCAN(can_devtype, can_devindex, can_canindex, can_baudrate, can_acccode, can_accmask)
        self.can.opendevice()
        self.can.initcan()
        self.can.startcan()
    '''
    while 1:
        if kbq(): break
        res=can.receive()
        for i in range(res):
            print(can.receivebuf[i])
            print(can.receivebuf[i].getdata())
        # sql.copy(can.receivebuf, can.receivenum, can.timeinterval)
        # sql.storage()
        # sql.commit()
        can.sendbuf[0].ID = 0x123
        can.sendbuf[0].DataLen = 8
        can.sendbuf[0].Data[0] = 0x00
        can.sendbuf[0].Data[1] = 0x11
        can.sendbuf[0].Data[2] = 0x22
        can.sendbuf[0].Data[3] = 0x33
        can.sendbuf[0].Data[4] = 0x44
        can.sendbuf[0].Data[5] = 0x55
        can.sendbuf[0].Data[6] = 0x66
        can.sendbuf[0].Data[7] = 0x77

        can.sendbuf[1].ID = 0x321
        can.sendbuf[1].setdata([1, 2, 3, 4, 5, 6, 7, 8])

        can.transmit(2)
    '''
    def CANClose(self):
        del self.can
    # del sql


def kbq():
    if msvcrt.kbhit():
        ret = ord(msvcrt.getch())
        if ret == 113 or ret == 81:  # q or Q
            return 1


if __name__ == "__main__":
    CAN_Config()
