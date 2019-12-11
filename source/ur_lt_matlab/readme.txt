05.12.2019 Finn Linzer - TU Wien
Im gleichen Ordner dieser "readme.txt" liegt das auszuführende .m-file "apply_trafo_leverarm.py", dieses muss in Matlab mit "run" gestartet werden.

Im Unterordner "data_add-leverarm-to-RF" liegt ein weiteres .m-file "data_trafo_input.m", dieses muss zusätzlich in Matlab geladen werden, in diesem .m-file werden die Messdaten der Messepoche eingelesen. 
2 Dateien müssen für die erfolgreiche Berechnung im richtigen Format eingelesen werden:
---> lt.txt         (oder LTbezugUR_ausSA.txt über SA) - enthält die gemessenen Koordinaten mit CCR in mm ! (wird erst später im Skript in meter ungerechnet)
Auszug:
1   2286.542190   -23.796892   52.287136
2   2153.737950   159.335274   81.135763
3   1954.453189   433.481556   -3.177771

---> juri.txt       (oder 191117_pose.txt durch MP_Server.py) - enthält die durch MP_Server.py aufgezeichneten POSEN (1 Punktnummer, 3 Koordinaten, 3 Winkelbeschreibungen, in m und rad).
Auszug:
1,-0.16378,-0.19230,0.56130,-0.08952,-0.98449,0.74977
2,-0.17611,0.00131,0.63473,-0.52375,-0.81440,0.97918
3,-0.19718,0.33355,0.62013,-1.14305,-0.49390,1.24778

(oder ähnlicher Name oder anderes Format (txt,...), kann aber ja entsprechend in matlab eingeladen werden)
Das vorliegende Skript ist bereits für die in AKIG benutzten Ausgabeformate (MP_Server.py und SA) optimiert und funktioniert für die am 27.11.19 aufgenommen Daten.


(Für data.R (etwa Zeile 23) und data.LT (etwa 3 Zeilen darunter) müssen die Werte ordentlich eingelesen werden, hierfür die Range entsprechend des eigenen Datentyps anpassen.)



Laut Sabine reichen 16 Messungen aus um in ein Ergebnis im Zehntel MM Bereich zu bekommen.

Um den Globaltest zu bestehen müssen die stochastischen Annahmen über den Roboter geändert werden (Anfahrgenauigkeit X,Y,Z).
Dafür den Wert in Zeile 75 (    adj.Cll_R = eye(3,3)*0.3e-03^2; % Kov ^2 %HIER A PRIO ANNAHME ÄNDERN GT) solange anpassen bis die Ausgaben 
nach dem "run" von "apply_trafo_leverarm.m" etwa
GT_True == 1 und GT_Apost~~1 
entsprechen.
Statt 0.3e-03 einen anderen Wert (z.B. 0.42e-03) verwenden und der Auswirkung entsprechend weiter optimieren.

Das Ergebnis liegt am Ende im Struct: adj
adj.x enthält 9 Werte | rx ry rz (LT zu UR) | X Y Z (LT zu UR) | dx dy dz (tool0 zu lever) --- alle Werte in Radiant und Meter
adj.stdX enthält die entsprechenden Standardabweichungen
