 
Das von Tomas erstellte File joints_trafo.txt enhält relativ unkorrelierte Posen die der Roboterarm UR5 anfährt. Von jeder
Pose aus kann der aufgesetzte CCR gemessen werden. Der Tracker steht dabei in etwa 6 Meter Entfernung zum Tisch in die Richtung, in
die die erste Pose zeigt (seitlich vom Tisch weg). Auch der CCR muss in diese richtung verdreht werden. 
Falls Kontrolle darüber herschen soll, wann der Roboterarm sich weiter bewegt und wann nicht, kann ein bool an
den Roslaunch befehl übergeben werden.
zB.:
$ roslaunch igros_ur move_juri.launch joints_file:=/home/finn/workspace_ur/src/igros_ur/trajectories/joints_trafo.txt speed_factor:=0.5 user_confirm_proceed:= true
