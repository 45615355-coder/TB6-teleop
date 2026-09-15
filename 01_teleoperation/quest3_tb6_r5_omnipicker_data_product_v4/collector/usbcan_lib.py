"""ZLG libusbcan driver - structures from usbcan.py, lib from /lib/libusbcan.so."""
from ctypes import byref, cdll
from pathlib import Path

import usbcan as _zlg

LIB_PATH = str(Path(__file__).resolve().parent / "vendor" / "omnipicker" / "libusbcan.so")

lib = cdll.LoadLibrary(LIB_PATH)

USBCAN_I = _zlg.USBCAN_I
USBCAN_II = _zlg.USBCAN_II
ZCAN_CAN_OBJ = _zlg.ZCAN_CAN_OBJ
ZCAN_CAN_INIT_CONFIG = _zlg.ZCAN_CAN_INIT_CONFIG
ZCAN_CAN_BOARD_INFO = _zlg.ZCAN_CAN_BOARD_INFO

BAUD_500K = 0x1C00
BAUD_1M = 0x1400


def _val(x):
    return x.value if hasattr(x, "value") else x


class CanDevice:
    def __init__(self, device_type=USBCAN_I, device_index=0, channel=0, baud=BAUD_500K):
        self.device_type = device_type
        self.device_index = device_index
        self.channel = channel
        self.baud = baud
        self._opened = False

    def open(self):
        lib.VCI_CloseDevice(self.device_type, self.device_index)
        if lib.VCI_OpenDevice(self.device_type, self.device_index, 0) == 0:
            raise RuntimeError("VCI_OpenDevice failed")

        info = ZCAN_CAN_BOARD_INFO()
        nch = 1
        if lib.VCI_ReadBoardInfo(self.device_type, self.device_index, byref(info)) == 1:
            nch = max(1, _val(info.can_Num))

        for ch in range(nch):
            cfg = ZCAN_CAN_INIT_CONFIG()
            cfg.AccCode = 0
            cfg.AccMask = 0xFFFFFFFF
            cfg.Reserved = 0
            cfg.Filter = 1
            cfg.Timing0 = self.baud & 0xFF
            cfg.Timing1 = self.baud >> 8
            cfg.Mode = 0
            ok_init = lib.VCI_InitCAN(self.device_type, self.device_index, ch, byref(cfg))
            ok_start = lib.VCI_StartCAN(self.device_type, self.device_index, ch)
            print("InitCAN(%d) %s" % (ch, "success" if ok_init else "fail"))
            print("StartCAN(%d) %s" % (ch, "success" if ok_start else "fail"))
            if not ok_init or not ok_start:
                raise RuntimeError("InitCAN/StartCAN(%d) failed" % ch)

        self._opened = True

    def flush_rx(self):
        if hasattr(lib, "VCI_ClearBuffer"):
            lib.VCI_ClearBuffer(self.device_type, self.device_index, self.channel)
        while True:
            frames = self.receive(max_count=100, wait_ms=10)
            if not frames:
                break

    def read_board_info(self):
        info = ZCAN_CAN_BOARD_INFO()
        if lib.VCI_ReadBoardInfo(self.device_type, self.device_index, byref(info)) != 1:
            return None
        return info.hw_Type(), _val(info.can_Num)

    def transmit_batch(self, can_id=0x100, count=10, send_type=0):
        msgs = (ZCAN_CAN_OBJ * count)()
        for i in range(count):
            msgs[i].ID = can_id
            msgs[i].SendType = send_type
            msgs[i].RemoteFlag = 0
            msgs[i].ExternFlag = 0
            msgs[i].DataLen = 8
            for j in range(8):
                msgs[i].Data[j] = j
        return lib.VCI_Transmit(
            self.device_type, self.device_index, self.channel, byref(msgs), count
        )

    def transmit(self, can_id, data, send_type=0):
        msg = ZCAN_CAN_OBJ()
        msg.ID = can_id
        msg.SendType = send_type
        msg.RemoteFlag = 0
        msg.ExternFlag = 0
        msg.DataLen = min(len(data), 8)
        for i in range(msg.DataLen):
            msg.Data[i] = data[i]
        return lib.VCI_Transmit(
            self.device_type, self.device_index, self.channel, byref(msg), 1
        )

    def receive(self, max_count=100, wait_ms=100):
        n = lib.VCI_GetReceiveNum(self.device_type, self.device_index, self.channel)
        if n == 0:
            return []
        if n > max_count:
            n = max_count
        buf = (ZCAN_CAN_OBJ * n)()
        rcv = lib.VCI_Receive(
            self.device_type, self.device_index, self.channel, byref(buf), n, wait_ms
        )
        return [(_val(buf[i].ID), bytes(buf[i].Data[: buf[i].DataLen])) for i in range(rcv)]

    def close(self):
        if not self._opened:
            return
        info = ZCAN_CAN_BOARD_INFO()
        nch = 1
        if lib.VCI_ReadBoardInfo(self.device_type, self.device_index, byref(info)) == 1:
            nch = max(1, _val(info.can_Num))
        for ch in range(nch):
            lib.VCI_ResetCAN(self.device_type, self.device_index, ch)
        lib.VCI_CloseDevice(self.device_type, self.device_index)
        self._opened = False

    def set_baud(self, baud):
        if self.baud == baud and self._opened:
            return
        self.close()
        self.baud = baud
        self.open()
