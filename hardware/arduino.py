from pyfirmata import util
import pyfirmata
import time
import sys


class Arduino(object):
    def __init__(self, port:str='COM3') -> None:
        self.board = pyfirmata.Arduino(port)
        self.iterator = util.Iterator(self.board)

    def digital(self, arduino_port:int, state:bool=True):
        """### Digital Function use for work with arduino Digital Pins

        Args:
            arduino_port (int): It is Conatining Port No of arduino Uno
            state (bool, optional): Define the state `True` or `False`. Defaults to True.

        ### Example :

        >>> import arduino, time

        >>> ard = arduino.Arduino()  # Default Arduino Port is COM3 and it is on windows machine.

        >>> for _ in range(10):
        ...     ard.digital(13, state=True)
        ...     time.sleep(1)
        ...     ard.digital(13, state=False)
        ...     time.sleep(1)
        """
        return self.board.digital[arduino_port].write(int(state))

    def exit(self):
        """
        ### Exit method : For Exiting from Arduino.
        """
        return self.board.exit()

    def pin_define(self, type:str, port:int, state:str):
        """### Pin Define Use For defining any Pin on your Arduino Board.

        Args:
            type (str): It's Conatin three moods are given below:
            - PWM      (for Pulse Width Modulation)
            - Analog   (for Defining Analog Pin)
            - Digital  (for Defining Digital Pin)

            port (int): For Defining Arduino Port Like `1`, `2` and etc...
            state (str): For Input or Output If `state='o'` then it is set on `Output Mood` and If `state='i'` then it is set on `Input Mood` or If `state='s'` then it is set on servo.

        ### Example :

            >>> import arduino, time
            >>> ard = arduino.Arduino()  # Default Arduino Port is COM3 and it is on windows machine.

            >>> pin_define(
                type='Analog',
                port=1,
                state=False
            )
        """

        types = ['pwm', 'analog', 'digital']
        type = type.lower()
        if type not in types:
            raise Exception("Mood Type not supported")
        else:
            type = type[0]

        port = int(port)

        return self.board.get_pin(f"{type}:{port}:{state}")

    def sleep(self, times:float):
        return time.sleep(times)

    def pprint(self, value:str) -> str:
        for i in str(value) + '\n':
            sys.stdout.write(i)
            sys.stdout.flush()
            self.sleep(.1)

    def arduinoTester(self, led_blink:int=10):
        """It is a simple program for testing arduino is working or not."""
        self.pprint("Connection Establishing succesfully...")
        for _ in range(led_blink):
            self.digital(13)
            self.sleep(1.)
            self.digital(13, False)
            self.sleep(1)

        return "Program is Executed..."


class Servo(Arduino):
    def __init__(self, port: str = 'COM3') -> None:
        super(Servo, self).__init__(port)

    def rotate(self, angle:int, pin:int) -> int:
        """### it is a Servo motor Rotator/Mover.

        Args:
            angle (int): It is angle to move the arduino.
            pin (int): Servo Motor Connected Pin.

        ### Example

        >>> import arduino, time

        >>> srv = Servo()   # Default COM3
        >>> srv.rotate(120, 9)
        """
        self.iter = self.iterator
        self.iter.start()
        self.pin_Servo = self.pin_define('Digital', port=pin, state='s')
        self.pin_Servo.write(angle)
