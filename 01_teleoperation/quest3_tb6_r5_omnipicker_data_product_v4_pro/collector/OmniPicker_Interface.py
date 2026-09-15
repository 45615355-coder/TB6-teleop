import time

from usbcan_lib import BAUD_1M, USBCAN_I, CanDevice, _val


class OmniPicker_Interface:
    """
    OmniPicker gripper interface for ZLG USBCAN on Linux (libusbcan.so).

    CAN downstream (6.3.1): D1=Pos, D2=Vel, D3=Force, D4=Acc, D5=Dec
    """

    def __init__(
        self,
        Can_ID=0x01,
        device_type=USBCAN_I,
        device_index=0,
        channel=0,
        baud=BAUD_1M,
    ):
        self.can_id = Can_ID
        self.channel = channel
        self.device_type = _val(device_type)
        self._can = CanDevice(device_type, device_index, channel, baud)
        self._can.open()

        info = self._can.read_board_info()
        if info:
            hw_type, can_num = info
            print(
                "Device: %s, CAN channels: %d, channel=%d, can_id=0x%X, "
                "device_type=%d, baud=0x%04X\n"
                % (hw_type, can_num, channel, Can_ID, self.device_type, baud)
            )
        print("Initialize Done!\n")

    @staticmethod
    def _scale(value):
        return int(max(0, min(100, value)) / 100 * 255)

    def _build_payload(self, pos, vel, force, acc, dec):
        return bytes(
            [
                0,
                self._scale(pos),
                self._scale(vel),
                self._scale(force),
                self._scale(acc),
                self._scale(dec),
                0,
                0,
            ]
        )

    def device_check(self):
        """Verify USB-CAN works (self-receive, no external bus needed)."""
        ret = self._can.transmit_batch(0x100, 10, send_type=2)
        ok = _val(ret) == 10
        self._can.flush_rx()
        return ok

    def read_feedback(self, timeout_ms=100):
        """Read latest upstream frame: (fault, state, pos%, vel%, force%)."""
        deadline = time.time() + timeout_ms / 1000.0
        last = None
        while time.time() < deadline:
            for can_id, data in self._can.receive():
                if (can_id & 0x1FFFFFFF) != self.can_id:
                    continue
                if len(data) >= 5:
                    last = data
            time.sleep(0.005)
        if last is None:
            return None
        return (
            last[0],
            last[1],
            last[2] / 255.0 * 100.0,
            last[3] / 255.0 * 100.0,
            last[4] / 255.0 * 100.0,
        )

    def control(self, Pos=0.0, For=100.0, Vel=100.0, Acc=100.0, Dec=100.0,
                flush_rx=True):
        """Send control frame only."""
        payload = self._build_payload(Pos, Vel, For, Acc, Dec)
        if flush_rx:
            self._can.flush_rx()
        ret = self._can.transmit(self.can_id, payload, send_type=0)
        if _val(ret) != 1:
            raise RuntimeError("VCI_Transmit failed (ret=%d)" % _val(ret))

    def open(self):
        self.control(Pos=100)

    def half_open(self):
        self.control(Pos=50)

    def close(self):
        self.control(Pos=0)

    def shutdown(self):
        self._can.close()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
