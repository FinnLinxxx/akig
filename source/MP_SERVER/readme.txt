05.12.2019 TU Wien - Finn Linzer

This is the work of Sabine Horvath!
This is a python2 project!

This is the server side to acquire angle states or poses of an UR5 Robot.
Accordingly the MP_Client must run on the robot (>>Load Program>>MP_Client.urp ...) while using. 
If you press "Stop Program" on the tablet (client side), the program exit.
If you press "continue" on the tablet (client side), the program send his angles and poses to the MP_server 
- please be aware that while data is acquired the robot is not moveable, while data is not acquired the robot is in freedrive mode.

comput1.py - is required for computation

You can run the MP_Server.py in PyCharm
or with linux inside the console
$ python MP_server.py
