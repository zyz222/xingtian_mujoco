import sys
import termios
import tty
import threading
import queue

class KeyboardController:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        self.stop_event = None

    def _get_key(self):
        """非阻塞获取按键"""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1) if not self.stop_event.is_set() else None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _input_loop(self):
        """键盘监听主循环"""
        default_vel = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        
        while not self.stop_event.is_set():
            key = self._get_key()
            if not key:
                continue

            # 命令处理
            cmd_map = {
                'q': ("quit", default_vel),
                '0': ("passive", default_vel),
                '1': ("stand", default_vel),
                '2': ("move", default_vel),
                '3': ("idle", default_vel),
                'w': (None, {"vx": 0.1, "vy": 0.0, "wz": 0.0}),
                's': (None, {"vx": -0.1, "vy": 0.0, "wz": 0.0}),
                'a': (None, {"vx": 0.0, "vy": 0.1, "wz": 0.0}),
                'd': (None, {"vx": 0.0, "vy": -0.1, "wz": 0.0}),
                'e': (None, {"vx": 0.0, "vy": 0.0, "wz": 0.1}),
                'c': (None, {"vx": 0.0, "vy": 0.0, "wz": -0.1})
            }

            # 获取当前指令
            if key.lower() in cmd_map:
                state_cmd, vel = cmd_map[key.lower()]
            else:
                print(f"无效的指令：'{key.lower()}'，请输入有效指令。")
                continue  # 如果是无效指令，跳过此轮循环

            # 如果命令是退出，设置停止事件
            if state_cmd == "quit":
                self.stop_event.set()
            
            # 处理命令并将其添加到命令队列
            if state_cmd or vel:
                current_cmd = state_cmd if state_cmd else self.command_queue.queue[0][0] if not self.command_queue.empty() else "passive"
                current_vel = {k: v + vel[k] for k, v in default_vel.items()} if vel else default_vel
                self.command_queue.put((current_cmd, current_vel))

                

    def start(self, stop_event):
        """启动键盘线程"""
        self.stop_event = stop_event
        self.thread = threading.Thread(
            target=self._input_loop,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        """停止键盘监听"""
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.5)