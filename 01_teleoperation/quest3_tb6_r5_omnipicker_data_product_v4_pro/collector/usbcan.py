"""ZLG libusbcan ctypes structures (from official usbcan_ii_libusb SDK)."""
from ctypes import Structure, c_byte, c_int, c_ubyte, c_uint32, c_uint8, c_ushort

USBCAN_I = c_uint32(3)
USBCAN_II = c_uint32(4)


class ZCAN_CAN_BOARD_INFO(Structure):
    _fields_ = [
        ("hw_Version", c_ushort),
        ("fw_Version", c_ushort),
        ("dr_Version", c_ushort),
        ("in_Version", c_ushort),
        ("irq_Num", c_ushort),
        ("can_Num", c_ubyte),
        ("str_Serial_Num", c_ubyte * 20),
        ("str_hw_Type", c_ubyte * 40),
        ("Reserved", c_ubyte * 4),
    ]

    def hw_Type(self):
        name = ""
        for c in self.str_hw_Type:
            if c > 0:
                name += chr(c)
            else:
                break
        return name


class ZCAN_CAN_INIT_CONFIG(Structure):
    _fields_ = [
        ("AccCode", c_int),
        ("AccMask", c_int),
        ("Reserved", c_int),
        ("Filter", c_ubyte),
        ("Timing0", c_ubyte),
        ("Timing1", c_ubyte),
        ("Mode", c_ubyte),
    ]


class ZCAN_CAN_OBJ(Structure):
    _fields_ = [
        ("ID", c_uint32),
        ("TimeStamp", c_uint32),
        ("TimeFlag", c_uint8),
        ("SendType", c_byte),
        ("RemoteFlag", c_byte),
        ("ExternFlag", c_byte),
        ("DataLen", c_byte),
        ("Data", c_ubyte * 8),
        ("Reserved", c_ubyte * 3),
    ]
