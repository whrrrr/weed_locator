#!/usr/bin/env python3
import argparse
import shlex
import socket
import struct
import sys
import time


CAN_EFF_FLAG = 0x80000000
REQ_ID = 0x601
RESP_ID = 0x581


def make_frame(can_id, data):
    return struct.pack("=IB3x8s", can_id, len(data), data.ljust(8, b"\x00"))


def parse_frame(frame):
    can_id, dlc, data = struct.unpack("=IB3x8s", frame)
    return can_id & ~CAN_EFF_FLAG, dlc, data[:dlc]


def le(value, size, signed=False):
    return int(value).to_bytes(size, "little", signed=signed)


class ZlacTool:
    def __init__(self, iface, timeout):
        self.iface = iface
        self.timeout = timeout
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.bind((iface,))
        self.sock.settimeout(timeout)

    def tx(self, can_id, data, note=""):
        print(f"TX {can_id:03X}#{data.hex().upper()} {note}".rstrip(), flush=True)
        self.sock.send(make_frame(can_id, data))

    def wait_sdo(self, index, sub):
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                can_id, dlc, data = parse_frame(self.sock.recv(16))
            except socket.timeout:
                break
            if can_id != RESP_ID:
                print(f"RX {can_id:03X}#{data.hex().upper()} ignored", flush=True)
                continue
            print(f"RX {can_id:03X}#{data.hex().upper()}", flush=True)
            if len(data) >= 4 and data[1] == (index & 0xFF) and data[2] == (index >> 8) and data[3] == sub:
                if data[0] == 0x80:
                    abort = int.from_bytes(data[4:8], "little")
                    raise RuntimeError(
                        f"SDO abort index=0x{index:04X}:{sub:02X} code=0x{abort:08X}"
                    )
                return data
        raise TimeoutError(f"timeout waiting for 0x{RESP_ID:03X} index=0x{index:04X}:{sub:02X}")

    def read(self, index, sub, typ):
        data = bytes([0x40, index & 0xFF, index >> 8, sub, 0, 0, 0, 0])
        self.tx(REQ_ID, data, f"read 0x{index:04X}:{sub:02X}")
        resp = self.wait_sdo(index, sub)
        raw = resp[4:8]
        if typ == "u8":
            return raw[0]
        if typ == "i8":
            return int.from_bytes(raw[:1], "little", signed=True)
        if typ == "u16":
            return int.from_bytes(raw[:2], "little", signed=False)
        if typ == "i16":
            return int.from_bytes(raw[:2], "little", signed=True)
        if typ == "u32":
            return int.from_bytes(raw[:4], "little", signed=False)
        if typ == "i32":
            return int.from_bytes(raw[:4], "little", signed=True)
        raise ValueError(f"unknown type {typ}")

    def write(self, index, sub, value, typ):
        if typ == "u8":
            cmd, payload = 0x2F, le(value, 1, False)
        elif typ == "i8":
            cmd, payload = 0x2F, le(value, 1, True)
        elif typ == "u16":
            cmd, payload = 0x2B, le(value, 2, False)
        elif typ == "i16":
            cmd, payload = 0x2B, le(value, 2, True)
        elif typ == "u32":
            cmd, payload = 0x23, le(value, 4, False)
        elif typ == "i32":
            cmd, payload = 0x23, le(value, 4, True)
        else:
            raise ValueError(f"unknown type {typ}")
        data = bytes([cmd, index & 0xFF, index >> 8, sub]) + payload.ljust(4, b"\x00")
        self.tx(REQ_ID, data, f"write 0x{index:04X}:{sub:02X}={value}")
        self.wait_sdo(index, sub)

    def status(self):
        vals = {
            "pos_l": self.read(0x6064, 0x01, "i32"),
            "pos_r": self.read(0x6064, 0x02, "i32"),
            "spd_l": self.read(0x606C, 0x01, "i32") / 10.0,
            "spd_r": self.read(0x606C, 0x02, "i32") / 10.0,
            "hall_l": self.read(0x2034, 0x01, "u16"),
            "hall_r": self.read(0x2034, 0x02, "u16"),
            "temp_l": self.read(0x2032, 0x01, "i16") / 10.0,
            "temp_r": self.read(0x2032, 0x02, "i16") / 10.0,
            "temp_drv": self.read(0x2032, 0x03, "i16") / 10.0,
            "vbus": self.read(0x2035, 0x00, "u16") / 100.0,
            "sync": self.read(0x200F, 0x00, "u16"),
            "mode": self.read(0x6061, 0x00, "i8"),
            "sw": self.read(0x6041, 0x00, "u32"),
            "fault": self.read(0x603F, 0x00, "u32"),
        }
        left_sw = vals["sw"] & 0xFFFF
        right_sw = (vals["sw"] >> 16) & 0xFFFF
        print(
            "STATUS "
            f"pos_l={vals['pos_l']} pos_r={vals['pos_r']} "
            f"spd_l={vals['spd_l']:.1f}rpm spd_r={vals['spd_r']:.1f}rpm "
            f"hall_l={vals['hall_l']} hall_r={vals['hall_r']} "
            f"temp_l={vals['temp_l']:.1f}C temp_r={vals['temp_r']:.1f}C temp_drv={vals['temp_drv']:.1f}C "
            f"vbus={vals['vbus']:.2f}V sync={vals['sync']} mode={vals['mode']} "
            f"sw_l=0x{left_sw:04X} sw_r=0x{right_sw:04X} "
            f"fault=0x{vals['fault']:08X}",
            flush=True,
        )

    def prep_pos(self, amps):
        tenths = int(round(amps * 10))
        print(f"Preparing position mode, current limit left/right={amps:g}A ({tenths} x 0.1A)")
        self.write(0x2015, 0x01, tenths, "u16")
        self.write(0x2015, 0x02, tenths, "u16")
        self.write(0x200F, 0x00, 1, "u16")
        self.write(0x6060, 0x00, 1, "i8")
        self.write(0x6083, 0x01, 1000, "u32")
        self.write(0x6083, 0x02, 1000, "u32")
        self.write(0x6084, 0x01, 1000, "u32")
        self.write(0x6084, 0x02, 1000, "u32")
        self.write(0x6081, 0x01, 1, "u32")
        self.write(0x6081, 0x02, 1, "u32")
        self.write(0x607A, 0x02, 0, "i32")

    def enable(self):
        self.write(0x6040, 0x00, 0x0006, "u16")
        self.write(0x6040, 0x00, 0x0007, "u16")
        self.write(0x6040, 0x00, 0x000F, "u16")
        self.status()

    def stop(self):
        self.write(0x6040, 0x00, 0x0006, "u16")
        self.status()

    def clear_fault(self):
        self.write(0x6040, 0x00, 0x0080, "u16")
        self.status()

    def jog(self, counts):
        counts = int(counts)
        before = self.read(0x6064, 0x01, "i32")
        print(f"JOG start pos_l={before} target_delta={counts}", flush=True)
        self.write(0x607A, 0x02, 0, "i32")
        self.write(0x607A, 0x01, counts, "i32")
        time.sleep(0.15)
        self.write(0x6040, 0x00, 0x004F, "u16")
        time.sleep(0.15)
        self.write(0x6040, 0x00, 0x005F, "u16")
        time.sleep(1.0)
        after = self.read(0x6064, 0x01, "i32")
        fault = self.read(0x603F, 0x00, "u32")
        print(f"JOG result pos_l={after} moved={after - before} fault=0x{fault:08X}", flush=True)

    def speed(self, rpm, seconds):
        rpm = int(rpm)
        seconds = float(seconds)
        self._speed_run(left_rpm=rpm, right_rpm=0, seconds=seconds, label="left")

    def speedr(self, rpm, seconds):
        rpm = int(rpm)
        seconds = float(seconds)
        self._speed_run(left_rpm=0, right_rpm=rpm, seconds=seconds, label="right")

    def _speed_run(self, left_rpm, right_rpm, seconds, label):
        print(
            f"SPEED start {label} left={left_rpm}rpm right={right_rpm}rpm duration={seconds}s",
            flush=True,
        )
        self.write(0x200F, 0x00, 0, "u16")
        self.write(0x6060, 0x00, 3, "i8")
        self.write(0x6083, 0x01, 1000, "u32")
        self.write(0x6083, 0x02, 1000, "u32")
        self.write(0x6084, 0x01, 1000, "u32")
        self.write(0x6084, 0x02, 1000, "u32")
        self.write(0x6040, 0x00, 0x0006, "u16")
        time.sleep(0.1)
        self.write(0x6040, 0x00, 0x0007, "u16")
        time.sleep(0.1)
        self.write(0x6040, 0x00, 0x000F, "u16")
        self.write(0x60FF, 0x01, left_rpm, "i32")
        self.write(0x60FF, 0x02, right_rpm, "i32")
        time.sleep(seconds)
        self.write(0x6040, 0x00, 0x0006, "u16")
        time.sleep(0.2)
        pos_l = self.read(0x6064, 0x01, "i32")
        pos_r = self.read(0x6064, 0x02, "i32")
        fault = self.read(0x603F, 0x00, "u32")
        mode = self.read(0x6061, 0x00, "i8")
        print(
            f"SPEED result pos_l={pos_l} pos_r={pos_r} mode={mode} fault=0x{fault:08X}",
            flush=True,
        )


HELP = """Commands:
  status              Read position/speed/hall/temp/vbus/state/fault.
  prep [amps]         Set safe position mode. Default amps=3.0. Does not move.
  enable              Run 6040: 06 -> 07 -> 0F. Does not command movement.
  jog <counts>        Relative left move, e.g. jog 100 or jog -100.
  stop                Send 6040=06 and read status.
  clear_fault         Send 6040=80 and read status.
  read IDX SUB TYPE   Example: read 6064 01 i32.
  write IDX SUB TYPE VALUE
                      Example: write 2015 01 u16 30.
  speed <rpm> [sec]   Low-speed left run for a short time, then stop.
  speedr <rpm> [sec]  Low-speed right run for a short time, then stop.
  help
  quit

Safe pattern for your current setup:
  status
  prep 3
  enable
  jog 100
  jog -100
  stop
"""


def parse_int(text):
    return int(text, 0)


def repl(tool):
    print(HELP, flush=True)
    while True:
        try:
            line = input("zlac> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        try:
            parts = shlex.split(line)
            cmd = parts[0].lower()
            if cmd in ("quit", "exit"):
                return
            if cmd == "help":
                print(HELP, flush=True)
            elif cmd == "status":
                tool.status()
            elif cmd == "prep":
                amps = float(parts[1]) if len(parts) > 1 else 3.0
                tool.prep_pos(amps)
            elif cmd == "enable":
                tool.enable()
            elif cmd == "stop":
                tool.stop()
            elif cmd == "clear_fault":
                tool.clear_fault()
            elif cmd == "jog":
                if len(parts) != 2:
                    raise ValueError("usage: jog <counts>")
                tool.jog(parse_int(parts[1]))
            elif cmd == "speed":
                if len(parts) not in (2, 3):
                    raise ValueError("usage: speed <rpm> [sec]")
                rpm = parse_int(parts[1])
                sec = float(parts[2]) if len(parts) == 3 else 2.0
                tool.speed(rpm, sec)
            elif cmd == "speedr":
                if len(parts) not in (2, 3):
                    raise ValueError("usage: speedr <rpm> [sec]")
                rpm = parse_int(parts[1])
                sec = float(parts[2]) if len(parts) == 3 else 2.0
                tool.speedr(rpm, sec)
            elif cmd == "read":
                if len(parts) != 4:
                    raise ValueError("usage: read IDX SUB TYPE")
                idx, sub, typ = parse_int(parts[1]), parse_int(parts[2]), parts[3]
                value = tool.read(idx, sub, typ)
                print(f"VALUE 0x{idx:04X}:{sub:02X} {typ} = {value}", flush=True)
            elif cmd == "write":
                if len(parts) != 5:
                    raise ValueError("usage: write IDX SUB TYPE VALUE")
                idx, sub, typ, value = parse_int(parts[1]), parse_int(parts[2]), parts[3], parse_int(parts[4])
                tool.write(idx, sub, value, typ)
            else:
                print(f"Unknown command: {cmd}. Type help.", flush=True)
        except Exception as exc:
            print(f"ERROR {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Interactive ZLAC8015D V4 CANopen SDO helper")
    parser.add_argument("--iface", default="can0")
    parser.add_argument("--timeout", type=float, default=0.4)
    args = parser.parse_args()
    try:
        repl(ZlacTool(args.iface, args.timeout))
    except OSError as exc:
        print(f"ERROR opening {args.iface}: {exc}", file=sys.stderr)
        print("Is can0 up? Example: ip link set can0 up", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
