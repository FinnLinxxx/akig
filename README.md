# akig
akig ROS Tutorial

Das Einloggen in den eigenen Linux-Account erfolgt über den normale Geo-Benutzernamen und das normale Geo-Passwort.
Die verfügbaren Geo-Computer können hierfür verwendet werden, mittels Remote Desktop Verbindung wird darauf zugegriffen.
Die IP lautet: 128.130.8.200:4889, keine weiteren Angaben als IP und Port sind notwendig.
Die Sicherheitsinformationen müssen bestätigt werden.

Erstmalig eingeloggt ist es sinnvoll sich für die Default Settings zu entscheiden. Anschließend muss die Tastatur auf Deutsch umgestellt werden.
Hierfür oben links auf Applications >> Settings >> Keyboard > Layout > "den Haken bei Use system defaults entfernen" > Add > "German" > Alle anderen
Layouts anklicken und mit Delete entfernen > Close > Fertig.

Um den Roscore in der Bash starten zu können ist es wichtig den Installationspfad dem eigenen System bekannt zu geben, hierfür:
$ source /opt/ros/kinetic/setup.bash

Damit dieser Vorgang nicht in jedem neuen Terminal/Bash erneut eingegeben werden muss kann dieser auch in die .bashrc (Resource Datei) geschrieben werden, hierfür nacheinander:
$ cd 
$ echo "source /opt/ros/kinetic/setup.bash" >> .bashrc

Einmal durchgeführt muss dies auch bei mehreren Log-Ins und Terminals nicht wiederholt werden.

Der Roscore kann anschließend gestartet werden, da wir mehrere Roscores auf einem System verwalten müssen ist es wichtig diesem einen eigenen Port zuzuwiesen, hier xxxxx durch
eine 5 stellige Zahl ersetzen:
$ roscore -p xxxxx
zusätzlich ist es nötig ros den Port in der Environment bekanntzugeben ($ env). Hierfür:
$ export ROS_MASTER_URI=http://192.168.178.217:xxxxx
Auch dieser Befehl kann der bash-Resourcedatei (.bashrc) beigebracht werden und muss daher anschließend auch nicht nochmal erneut aufgerufen werden, hierfür nacheinander:
$ cd
$ echo "export ROS_MASTER_URI=http://192.168.178.217:xxxxx" >> .bashrc

--- 

Anschließend kann ein rosbags abgespielt werden:
$ rosbag play rotateBaseBFspeed05Pts125hzLAN.bag
Ein python Program, das dem Datenfluss zuhört kann zum beispiel mit
$ python trackerlistener_example.py gestartet werden.

Hierfür müssen sie sich im passenden Pfad befinden oder diesen korrekt für .py angeben.
$ cd
$ akig/ROS_Tutorial/listenerProgram


Unbedingt daran denken, falls der roscore geschlossen worden sein sollte (strg+C), dass dieser wieder mit dem gewählten Port gestartet wird.
$ roscore -p xxxxx
