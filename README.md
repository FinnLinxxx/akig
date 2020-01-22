# akig

Befehle für Übung am 22.01.2020

### git repo aktualisieren
```bash
$ cd akig
$ git pull
```

### startup roscore
```bash
$ roscore -p [PORTNR]
```

Die createGlobalPtclFromASC.py nimmt eine bestehende ASC Datei und streamt diese als Punktwolke ins ROS
Um zu funktionieren muss der Dateiname.asc, time_delta, parent_frame und die start_meas_tscan_rostime angegeben werden, die dem ermittelten ROS-Zeitstempel aus der Zeitsynchro entspricht. Gegebenfalls müssen auch die Achsen angepasst werden. 
Zusätzlich muss ein neuer Algorithmus gefunden werden, der die Punkte aus der ASC in der Zeitauflösung exakter ans ROS übergibt. Dafür muss auch auf die Performance geachtet werden. Wenn jeder Punkt einen einzelnen Zeitstempel besitzen sollte, wird es eventuell zu anspruchsvoll diese auch als Einzelpunkte zu transformieren, vor allem vor dem Hintergrund, dass für den Roboterarm die Pose mit nur 125hz gemessen werden kann.

```bash
$ cd akig/source/global_tscan_ptcl/publishASCfromTScanCenter
$ python3 createGlobalPtclFromASC.py
```

Bevor wir mit dem Rosbag arbeiten können setzen wir den Roscore auf simulation, damit wir einen eigenen Zeitbezug aufstellen können. Dieser Befehl muss nach dem Start eines roscores ausgeführt werden.

```bash
$ rosparam set use_sim_time 1
$ rosbag play rasp_speed_03_17_01.bag --clock
```

Nun folgt die Transformation mit den hier gezeigten Übergabeparametern:
```bash
$ rosrun tf_points_global transform_point2pointcloud _ptcl2_global_frame:=map _ptcl2_local_frame:=leverarm _ptcl2_input_topic:=/tscan_cloud2 _ptcl2_output_topic:=/tscan_cloud2_global _drop_when_same_position:=false
```

Eine einzelne Pointcloud2 Punktwolke kann in ROS mit folgendem Befehl gespeichert werden, diesen am besten in einem eigenen Unterordner ausführen
```bash
$ rosrun pcl_ros pointcloud_to_pcd input:=/tscan_cloud2_global
```

Alle entstandenen PCD-Files (lesbares Format für Punktwolken) können mit folgendem Linux Befehl zur ptcl_as_ascii.txt zusammengefasst werden, diese Datei kann so zb in Cloudcompare angeschaut werden.

```bash
$ printf '%s\n\n' "$(tail -n +30 *.pcd)" > ptcl_as_ascii.txt
```




















---
---
---
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

Hallo hier ist Markus
