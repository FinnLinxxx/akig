 
# AK IG

Für die LVA Ausgewählte Kapitel der Ingenieurgeodäsie (128.033) wird im Wintersemester 2019 die Entwicklung eines eigenen 3D Messsystems untersucht. Sensorik aus dem Bereich der Automation und flächenhaften Erfassung wird im geodätischen Kontext entwickelt.

Weitere Informationen sind dem Beschreibungstext in [TISS](https://tiss.tuwien.ac.at/course/courseAnnouncement.xhtml?dswid=8545&dsrid=730&courseNumber=128033&courseSemester=2019W) zu entnehmen:

### Merkmale
Semesterwochenstunden: 2.0
ECTS: 3.0
Typ: VU Vorlesung mit Übung
### Lernergebnisse
Nach positiver Absolvierung der Lehrveranstaltung sind Studierende in der Lage...

* den Bewegungsablauf eines Roboterarms aufgabenbezogen zu entwerfen und die Pose mittels Vorwärtskinematik zu berechnen.
mit einem Lasertracker eigenständig hochgenaue Daten für die weitere Verarbeitung zu bestimmen.
Möglichkeiten zur Ansteuerung geodätischer Messtechnik und Roboter auch aus dem Bereich open-source zu benennen und zu beurteilen.
* Ideen zur Verknüpfung von Daten mit eigenen Prozessanweisungen zu formulieren und umzusetzen.
* eigenes Wissen durch selbständiges Arbeiten auf Problemstellungen modernster Ansätze zu übertragen.
* räumliche Daten der Industriemesstechnik als Messergebnis auf eine Fragestellung hin zu analysieren und zu präsentieren.

### Inhalt der Lehrveranstaltung
Vorstellung und Diskussion von Methoden und Sensoren für die raumkontinuierliche geometrische Bestimmung. Für die Bestimmung der Geometrie kommen terrestrische Laserscanner und Methoden der Industrievermessung (LaserTracker+T_Scan) zum Einsatz. Gruppen erarbeiten und präsentieren einzelne Arbeitspakete die zusammenhängend als übergeordnete Aufgabe die Erzeugung qulitativ hochwertiger Punktwolken haben. Anhand dieser Punktwolken können systematische Einflüsse der Messmethodik aufgedeckt werden. Die Programmierkenntnisse der Studierenden werden in den Übungen gefördert und die Umsetzung der einzelnen Aufgaben unterstützt. Die Studierenden gewinnen ein Einblick darüber was in der Verbindung von Programmierung und Geodäsie möglich ist.

### Methoden
Einführung in die proprietäre Software RoboDK (Roboterarm) und Spatial Analyzer (Lasertracker). Begleitetes Tutorial für die Anwendung des (open-source) Robot Operating Systems (ROS). Unterstützung bei der Programmierung in Python/Matlab zur Umsetzung möglicher Ideen und Ansätze.

### Prüfungsmodus
Prüfungsimmanent

Kombination von Vorlesungs-, Übungs- und Diskussionseinheiten inklusive selbständiger Umsetzung von Konzeptideen (Laserscanner, Lasertracker+T-Scan, MatLab/Python).

### Vortragende
[Neuner, Hans-Berndt](https://tiss.tuwien.ac.at/adressbuch/adressbuch/person/277368)

[Linzer, Finn](https://tiss.tuwien.ac.at/adressbuch/adressbuch/person/323432)

E120 Department für Geodäsie und Geoinformation

---

### RoboScan

RoboScan ist ein neu zu entwickelndes System bei dem die T-Scan nicht im Tracker-Zusammenhang arbeitet, sondern ihre Pose über einen Roboterarm (UR5) bezieht. 

Zur Verfügung stehen ein:

* [Linux Cheat Sheet](https://github.com/FinnLinxxx/akig/blob/master/ROS_Tutorial/LINUX_Cheat_Sheet.pdf)

* [ROS Cheat Sheet](https://github.com/FinnLinxxx/akig/blob/master/ROS_Tutorial/ROS_Cheat_Sheet.pdf)

* [GIT Cheat Sheet](https://github.com/FinnLinxxx/akig/blob/master/ROS_Tutorial/GIT_Cheat_Sheet.png)


## Aufnahme Punktwolke T-Scan

Den [T-Scan Spezifikationen](https://w3.leica-geosystems.com/downloads123/m1/metrology/t-scan/brochures/leica%20t-scan%20brochure_de.pdf) entsprechend können Punkte flächenhaft mit einer Genauigkeit besser 0,1 mm erfasst werden. Für uns wichtig ist, dass die T-Scan vom Tracker unabhängig betrieben werden kann. Der eigentliche Hersteller, der die Sensorik des Zeilenlaserscanners entwickelt, hat nennt sich Steinbichler, diese Firma wurde 2015 von ZEISS übernommen. Für den von uns gedachten Einsatz ist ebenfalls wichtig, dass das zuvor handgehaltene System am Roboterarm mit einem Schnellspanner (örtlich quasi wiederhellstellbar) befestigt werden kann. Das Datenkabel der T-Scan selbst ist nur sehr kurz, ein Verlängerungskabel erhöht die Reichweite. Auf der anderen Seite muss dieses auf der Vorderseite in den SCANNER CONTROLLER (roter Strich zu rotem Strich) gesteckt werden (Aufschrift: SCANNER). Der Probe Anschluss ist für die Kommunikation mit dem LasertrackerController gedacht, ob dieses Kabel für unseren Zweck gebraucht wird ist derzeit nicht bekannt. Auf der Rückseite des SCANNER CONTROLLERS ist das Kabel des LAN- und des Console-Anschlusses an den Betriebs-PC anzuschließen. Ob das Clock/Trigger Kabel zum LasertrackerController hin angeschlossen sein muss, ist nicht bekannt. Insgesamt ist der Aufbau bereits erfolgt (bis auf eventuell das Einstecken der T-Scan selbst), falls dies nicht der Fall ist nehmen sie bitte kontakt auf.

Eine Verknüpfung names Scanner Controller (gelbliches Laptop Symbol) liegt auf dem Desktop, dabei handelt es sich um ein Konsolenprogramm mit dem Namen Teraterm, dieses kann auch über den eigentlichen Pfad `(C:\Program Files (x86)\teraterm\ und dann ttermpro.exe)` gestartet werden.
Das Konsolenprogram muss gestartet werden bevor der Strom am eigentlichen SCANNER CONTROLLER auf "ein" geschaltet wird. Sollte es zu einem Fehler kommen, z.B. das COM3 nicht gefunden werden kann, empfiehlt sich ein neustart des Computers. Bei funktionierender Kommunikation werden mehrere Ausgabeparameter in der Anzeige ausgegeben, bevor nach etwa 15 Sekunden ein vorwiegend grün leuchtendes Fenster erscheint. In diesem Fenster können nun Befehle an das T-Scan System durch einfache Tastatureingabe übergeben werden. Nach dem erfolgreichen Hochstarten könnte die Befehlskette wie folgt aussehen:

Um das System bzw. die Ausgabedatei auf ASCII-Symbolik umzuschalten
```
asc
(enter)
```
Im Fenster erscheint daraufhin unter dem Punkt Storage Mode: ASCII (oder ähnlich)

Mit dem folgenden Befehl können (für etwa 30 Sekunden bzw. 15 MB) Scannerzentrische Daten aufgenommen werden.
```
smo
(enter)
```
Der folgende Befehl beendet die Aufnahme
```
smf
(enter)
```

Die Daten liegen daraufhin im Ordner: `C:\ScannerController` mit dem laufenden Namen SCAN000*.asc vor und können mit einem einfachen Textverarbeitungsprogramm eingsehen werden.

### Known-Problems/Todo
Hochladen Steinbichler PDF

### Ansätze

## Aufnahme Roboterarm Verfahrweg

`Achtung! Der Betrieb des Roboterarms setzt bestimmtes Wissen und stete Vorsicht voraus, da ansonsten Material oder Menschen zu schaden kommen könnten!`

`Der folgende Text zum Roboterarm wurde aus der Gruppenarbeit von Jansky, Mikschi, Konturek und Pitsch übernommen`

### Der Roboter UR5

#### Aufbau
Der Industrieroboter UR5 der dänischen Firma Universal Robots hat eine maximale Traglast von 5kg.
Seine Reichweite beträgt 850mm. Er hat sechs Gelenke, die sich alle um 720° bewegen lassen:
Das erste Gelenk wird „Base“ genannt und befindet sich am Fuß des Roboters.
Darauf folgen die „Shoulder“ und der Elbow. Vor dem Flansch befinden sich drei „Wrist {1,2,3}“-
Gelenke, deren Nummerierung auf der Seite der Basis beginnt.
>Der Roboter ist per Kabel mit seinem Steuergerät verbunden; drei weitere Kabel führen ans
Stromnetz, an einen WLAN-Router im IG-Messnetz und an ein „Tablet“ zum Bedienen des Roboters.

#### Sicherheitshinweise
Achtung, der Roboter kann Geschwindigkeiten von bis zu 1m/s erreichen (höhere Geschwindigkeiten
wurden deaktiviert). Es ist immer genug Abstand zum Roboter einzuhalten. Ebenfalls ist darauf zu
achten, dass der Roboter weder mit Hindernissen (z.B. der Tischplatte) noch mit sich selbst kollidiert.
In diesem Fall wird das Programm abgebrochen und der Roboter ist neu zu initialisieren.
Die Notaus-Taste am Tablet muss vor einer Kollision gedrückt werden!

#### Roboterflansch
Auf dem Flansch ist ein Schnellverschluss montiert, mittels dem verschiedene Zielobjekte auf den
Roboter montiert werden können. Mögliche Ziele sind:
>- Nest für einen Red Ring Reflector (RRR), eine Kugel mit 1,5“ Durchmesser von Leica (siehe
Fig. 1 – 3). Dabei wird der RRR mittels eines Magneten, der im Nest mit Federn gehalten
wird, am Nestaufsatz befestigt.
>- Die Leica T-Probe mit einer Spitze zum Messen von Punkten, auch schwer zugängliche an
Bauteilen. Die Spitze besteht aus hartem Glas und hat ca. 6mm Durchmesser.

#### Bedienung
Der Roboterarm kann entweder mittels des angehängten Tablets, durch die Software RoboDK oder wie später beschrieben wird auch mit ROS betrieben werden. RoboDK nimmt hierbei eine wichtige Rolle ein, indem abzufahrende Pfade im vorfeld geplant und anschließend in ROS genutzt werden können. Mit dem Tablet können nur einfache Bewegungen von Gelenken
steuern, wohingegen eine Programmierung von mehreren Bewegungsabläufen in RoboDK
realisierbar ist.
>Am Tablet wird der Roboter eingeschalten, die Motoren und das System mit dem RoboterBedienprogramm „PolyScope“ fahren hoch. Nach einer Initialisierung sind die Roboterarme zu starten. Dabei werden die Bremsen in den Gelenken aktiviert, was zu einem knacken führt und zu einer Bewegung des Roboters um bis zu einem Zentimeter. Erst jetzt hält der Roboter Lasten an ihrer
Position.

![ccrphoto](https://github.com/FinnLinxxx/akig/blob/master/manual/img/ccrFigure.png)

#### Koordinatensysteme
Der Roboter besitzt zwei KOS. Der Ursprung des ersten liegt genau in der Mitte jener Fläche, auf der
er montiert ist. Das hauptsächlich verwendete Koordinatensystem (KS) ist das des TCP, des ToolCenter Points. Es hat seinen Ursprung in der Mitte des Flansches.
>Die Position (bestehend aus drei Translationen und drei Rotationen) des TCP ist einstellbar.
Sofern für die Messung ein Punkt mit konstanter Entfernung zum Flanschmittelpunkt verwendet wird
(wie etwa die Spitze der T-Probe oder der RRR) sollte dieser Differenzvektor in Bezug auf die
Flanschmitte eingemessen werden. Durch die Bestimmung dieses Differenzvektors wird der TCP neu
festgelegt. Zu finden ist diese Funktion am Tablet unter Setup/neuen TCP erstellen. Dort wird
mehrere Male eine spezielle Vorrichtung (siehe Fig. 4) mit dem RRR angefahren, wobei darauf zu
achten ist möglichst viele verschiedene Winkelstellungen der Gelenke dafür zu benutzen. Im
Anschluss muss nur noch die „Set“-Taste gedrückt werden und der neue TCP ist im System
gespeichert. 

Bei Befestigung eines Werkstückes sollte nach Möglichkeit dessen Schwerpunkt als TCP angegeben
werden, sodass das Werkstück in allen Positionen und Ausrichtungen am besten gehalten werden
kann. Das geschieht durch feines Entgegenhalten der Motoren in den Gelenken des Roboters.
Zu diesem Zweck kann im Einstellungsmenü am Tablet des Roboters eingesehen werden, wie viel
Strom (in Ampere) jedes Gelenk in Ruhe als auch während der Ausführung eines Programms gerade
benötigt.

### Simulations- und Bediensoftware „RoboDK“

RoboDK ist ein proprietäres Computerprogramm zur Simulation von Bewegungen von Robotern
vieler verschiedener Hersteller, unter anderem Universal Robots. Vorab wurde der Roboter UR5 ins
Programm geladen, inklusive der beiden Koordinatensysteme (KOS) von Basis und Tool (bzw. Target)
Center Point (TCP).
#### Verbindungsaufbau 
Nach derzeitigem Stand der Technik, kann der UR5 bereits direkt von RoboDK aus angesteuert
werden. Dafür muss eine Verbindung im LAN „IG-Messnetz“ hergestellt werden. Wie in
nachfolgender Abbildung ersichtlich, ist der UR5 unter der IP-Adresse 192.168.178.5 und dem Port
(Roboter Anschluss) 50001 erreichbar. Mit einer ping-Anweisung wird die Verbindung überprüft.
Der Knopf „Get Position“ fragt die aktuelle Position des Roboterarmes ab und stellt sie dar.

![robodk](https://github.com/FinnLinxxx/akig/blob/master/manual/img/screenshot_RoboDK.png)

#### Programmoberfläche
Im linken Balken in der Grafik sind gelistet: Basis, Roboter UR5 und anschließend alle von ihm
erreichbaren Ziele (samt deren KOS). Darunter befinden sich die ausführbaren Programme mit ihren
jeweiligen „moves“. Diese bestehen aus einer Abfolge der anzufahrenden Punkte.
>Ein Doppelklick auf den Roboter UR5 links oben öffnet das UR5 Panel, das die ganze rechte Seite
einnimmt und vielerlei Eingaben ermöglicht. Speziell interessant ist das unterste Feld mit der
Überschrift „Achswinkel“. Dort werden die Winkelstellungen aller Gelenke angezeigt. In der bunten
Zeile darüber kann der Bezug des TCP (bzw. Werkzeugrahmen) zum Basis-KOS als Pose manipuliert
werden. Eine Pose besteht aus 6 Parametern: 3 für die Lage (Translation in [x,y,z]) und 3 für die
dortige Orientierung [Rx, Ry, Rz]. Es ist dabei auf die Einheit der Winkel zu achten (Grad oder
Radiant). Ebenso kann hier die Orientierung des TCP und ob eine Bewegung zu Kollisionen führt
erprobt werden.

#### Erstellen von Programmen
Für einen Bewegungsablauf am Roboter wird in UR5 ein Programm geschrieben, das man auch als
Endlosschleife laufen lassen kann. Beim Erstellen eines Programmes (in Fig. 5 ist Prog1 zu sehen,
durch Klicken auf den Button im Menüband) wird zuerst ein Bezugssystem definiert. Optional können
auch Geschwindigkeitsbeschränkungen für lineare Bewegungen oder für Rotationen festgelegt
werden.

#### Erstellen eines Punktes
Um einen neuen Punkt (target) zu erstellen kann durch Doppelklick auf die UR5-Base das Fenster
(wie in Fig. 5 auf der rechten Seite zu sehen) zum Verändern von Positionen aufgerufen werden. In
diesem Fenster können die Winkel der einzelnen Gelenke verändert werden. Diese
Winkeleinstellungen können anschließend einem Punkt zugeordnet werden (teach actual position),
wodurch die Position dieses Punktes im Koordinatensystem festgelegt ist.

#### Bewegung zu einem Punkt
RoboDK kennt zwei Arten für das Erreichen eines Punktes. Die empfohlene und bisher einzige
verwendete Methode ist „movej“ für „movejoints“, ihr Icon ist erkennbar am geschwungenen Weg
zwischen den Zielen. Dabei werden die Punkte durch gegebene Gelenkstellungen beschrieben und
der Weg zwischen ihnen als Differenz der Gelenkwinkel gefahren.
Die andere Funktion „move“ steht für eine Linearbewegung zum Ziel. Dabei werden die Gelenke so
bewegt, dass das Werkstück entlang einer Linie bewegt wird. Dies führt zu unterschiedlichen
Geschwindigkeiten der Gelenke im Laufe des Fahrweges.

>Vor dem Erstellen eines neuen „moves“ müssen die zu verwendeten „targets“ vorab definiert
worden sein. Sobald ein „target“ verändert wird, müssen auch alle „moves“, die dieses „target“
verwenden, neu erstellt werden.

#### Darstellung und Konfigurationen der Punkte
Durch Doppelklick auf den Roboter im Fenster können im UR5 Panel rechts unten die Positionen der Gelenke eingesehen werden (diese können auch manuell bewegt werden). Das Menü „Andere Konfigurationen“ zeigt meist zwischen 10 und 36 verschiedene
Möglichkeiten an, in welchen Arten der Punkt erreicht werden kann. Bei einigen Kombinationen
unterscheidet sich nur ein Gelenk um 360°. Entscheidend ist hier aber, ob der Ellbogen des Roboters
über oder unter dem TCP liegt und wie die Wrists zueinanderstehen. Für einen Einzelpunkt kommen
mehrere Konfigurationen in Frage, aber je komplexer die Fahrmuster werden, desto eingeschränkter
die Wahlmöglichkeit – vor allem dann, wenn die montierte T-Probe den Sichtkontakt zum
Lasertracker nicht verlieren darf.

>Um die Gelenkstellungen von Punkten aus der Liste links zu sehen, muss auf diese Punkte ein
Doppelklick erfolgen. Die Animation des Fahrweges entfällt und es wird die Situation am Ziel
angezeigt (mit Gelenkstellungen wie eingegeben). Bei einem einfachen Klick hingegen simuliert
RoboDK lediglich das Anfahren der gewünschten Position und Orientierung, meist auf kürzestem
Weg. Nicht selten kommt es dabei vor, dass „unmögliche“ Wege entstehen, die nicht gefahren
werden können. Der Roboter…

>• fährt zu Positionen unterhalb der Tischplatte oder

>• bewegt seine Gelenke so, dass er sich selbst durchfahren würde.

#### Kollisions-Detektion
RoboDK bietet den Vorteil, Kollisionen eines beliebigen Roboterteils mit
anderen Teilen zu detektieren. Ist diese Option eingeschaltet, so leuchtet ein grünes Häkchen vor
dem Nuklear-Symbol in der oberen Leiste. Bei einer Kollision wird die Animation des Fahrprogramms
gestoppt und beide kollidierenden Teile werden rot eingefärbt.
>Wenn alle Fahrmuster mit ihren Targets eingestellt und ausreichend in der Simulation getestet
wurden, kann mit dem echten Roboter gearbeitet werden. Nach Knopfdruck von „PTP Bewegung
ausführen“ leitet RoboDK die Befehle an den UR5 weiter.

#### Erzeugen einer TXT Roboterarm Bewegungsdatei

Vor dem eigentlichen Ansteuern des Roboterarms muss eine Textdatei erstellt werden, in der die Bewegungsabläufe sequenziell eingespeichert vorliegen. Eine einzelne Reihe beschreibt eindeutig die jeweils anzusteuernde Winkelposition aller 6 Schrittmotoren bzw. die anzufahrende Pose. Mehrere Reihen untereinander beschreiben eine Robotermessfahrt von Pose zu Pose. Diese Datei kann mit Hilfe von RoboDK erzeugt werden, da der Verfahrweg zwischen den Posen ohnehin dort auf Validität geprüft werden beziehen wir auch gleich die anzufahrenden Winkelpositionen darauß.
`Keine Roboterfahrt darf durchgeführt werden, wenn die Machbarkeit nicht zuvor in RoboDK auf mögliche Kollisionen geprüft worden ist!`

Wie eine solche Textdatei mit Hilfe von RoboDK erzeugt werden kann, habe ich einen einem Screencast auf TUWEL [hochgeladen](https://tuwel.tuwien.ac.at/mod/resource/view.php?id=712117) ([link zum TUWEL Kurs](https://tuwel.tuwien.ac.at/course/view.php?id=20221)). Es ist durchaus möglich die oben benannte Verbindung zum Roboterarm in RoboDK aufzubauen, den Arm per Freedrive zu verschieben um anschleßend die aktuelle Position zu erfragen und in RoboDK anzuzeigen, um diese dann als Target abzuspeichern. Dies kann den Prozess zum Aufbau einer eigenen TXT Datei beschleunigen. Zu beachten ist, dass bei dem Verfahrweg auch die aufgesetzte T-Scan mit bedacht werden muss.

### Durchführen einer Roboterfahrt in ROS

Die Verbindung zum eigenen Linux-Nutzer kann auf den im Labor befindlichen Computern in ähnlicher Weise über Remote Desktop durchgeführt werden, wie bereits im ROS-Tutorial gezeigt. Dafür muss man sich an einen der Computer anmelden und das Program für die Remote Desktop Umgebung starten, die Verbindungs-IP lautet: `128.130.8.200:4889`, Nutzer und Passwort entsprechen der Domäne. Die virtuelle Linux Umgebung befindet sich im gleichen IP-Adressbereich wie alle anderen Sensoren der Ingenieurgeodäsie, daher können von hier aus auch der Lasertracker oder Roboterarm angesteuert werden. 

Um die Verbindung zu testen, eignet es sich einen Ping auf den eingeschalteten Roboterarm durchzuführen, dieser hat die feste IP 192.168.178.5, öffnen sie also ein Konsolenfenster (Strg + Alt + T, ö.ä.) und führen sie folgenden Befehl aus.
```bash
$ ping 192.168.178.5
```
Nur wenn eine zufriedenstellende Antwort (latenzy < 100ms) zurück erhalten, kann in den folgenden Schritten eine stabile Verbindung zum Roboter aufgebaut werden.

Sie starten in einem weiteren Konsolenfenster den Roscore auf einem von ihnen gewählten Port mit 5 Ziffern (z.B. 12555)
```bash
$ roscore -p 12555
```
Damit sich alle weiteren ROS Programme von Ihnen auch auf diesen Port beziehen müssen Sie dies der Linux-Environment mitteilen, alle aktuellen environment Einstellungen können in Linux unter 
```bash
$ env
```
abgefragt werden, eine ganze Liste voll tut sich dabei auf. Um den aktuellen Port für ihr Linux System zu erfahren können sie die Liste nach bestimmten krtieren mit dem Program `grep` filtern, wir pipen `|` dabei den zuvorigen Listeninhalt und führen das Program `grep` nach einem Suchbegriff hin aus.
mitteilen, alle aktuellen environment Einstellungen können in Linux unter 
```bash
$ env | grep ROS_MASTER_URI
```
Ist der daraufhin angezeigte Port nicht ihrer (12555), müssen sie den neuen environment wert an das Linux System übergeben
```bash
$ export ROS_MASTER_URI=http://192.168.178.217:12555
```
Nun sollte der Wert stimmen. Damit sie nicht jedes Mal wenn sie eine neue Konsole starten diese Umgebungseinstellung tätigen müssen sollte die BASH-Resourcedatei `~/.bashrc` angepasst werden. Diese können sie z.B. mit dem Programm nano bearbeiten.
```bash
$ cd ~
$ nano .bashrc
```
Dort können sie in der letzten Zeile `export ROS_MASTER_URI=http://192.168.178.217:12555` hineinschreiben, dann wird dieser Befehl automatisch bei jedem Konsolenstart durchgeführt. An der gleichen Stelle können sie auch den Port ändern, nur müssen sie dann daran denken, dass eine Änderung (warum auch immer) erst dann aktiv wird, wenn sie eine neue Konsole starten.

Arbeiten wir alle auf verschiedenen Ports, kommen sich die ROS Projekte nicht gegenseitig in die quere.

Die für die Robotearmsteuerung benötigten Programme wurden von mir in ROS bereits kompiliert und somit für alle anderen auf den System auch ausführbar. Das Entwickeln und Aufbauen von ROS-Treibern zur Steuerung des Roboterarms oder des Lasertrackers sind zwar auch Teil der Arbeit der Ingenieurgeodäsie, jedoch würde dies den Rahmen der LVA übersteigen, daher nutzen wir die bisher erzeugte Infrastruktur. Wenn sich jemand von euch dafür interessiert, dann sprecht mich einfach darauf an. 

Ähnlich wie bei dem Export des Ports, sagen wir dem eigenen Linux-System wo es den Treiber für den Roboterarm finden kann, damit wir diesen benutzen können. Dieser befindet sich unter meinem Benutzernamen `flinzer` daher ist es wichtig, dass ihr genau diesen Pfad wählt. Damit das klappt könnten wir abermals in jeder neuen Konsole das Setup "sourcen"
```bash
$ source /home/flinzer/Workspace/devel/setup.bash
```
oder eben diesen Befehl (analog zum Port) in die `.bashrc` schreiben. Hierfür die Datei erneut öffnen und den Befehl dort hinschreiben, sodass er automatisch bei jedem Konsolenstart ausgeführt wird. Das ganze sieht dann zum Beispiel so aus:


```
...
export ROS_MASTER_URI=http://192.168.178.217:12555
source /home/flinzer/Workspace/devel/setup.bash
...
```

Anschließend sind wir in der Lage den einprogrammierten Bewegungsablauf an den Roboterarm zu übergeben und diesen kontrolliert ablaufen zu lassen. Übergebt eure eigene TXT Bewegungsdatei an das Skript, die TXT Datei muss irgendwo in eurem Verzeichnis `myname` abgelegt werden, z.B.:

```bash
$ roslaunch igros_ur move_juri.launch joints_file:=/home/myname/trajectories.txt speed_factor:=0.3
```
Der Speedfaktor gibt in Prozent an wie schnell sich der Roboter gegenüber der Maximalgeschwindigkeit bewegen soll. Vor allem bei ersten Tests, sollte die Geschwindigkeit nicht zu hoch gewählt werden.

Das Skript wird vor dem eigentlichen Verfahren anfragen, ob dies jetzt gewünscht ist (y/n). An dieser Stelle kann y ausgewält werden, um zu Verfahren oder es kann auf die weitere Interaktion mit dem Skript verzichtet werden, Winkelstelldaten und alles weitere werden jetzt schon an das ROS System übergeben.
Siehe:
```bash
$ rostopic list
```

Wichtig ist noch in einer weiteren Konsole das Anhängen des Roboterarmsystems (base_link) an das übergeordnete System (map). 

```bash
$ rosrun tf static_transform_publisher 0 0 0 0 0 0 map base_link 300
```

Sowie das anhängen des Leverarms an den tool0-Flansch
```bash
$ rosrun tf static_transform_publisher 0.02 0 0.33 0.4 0 -0.78539816339 tool0 lever 300
(oder ähnlich)
```

Mit rviz lässt sich das ganze visualiseren, hierbei müssen die TFs visualisiert werden, wenn noch nicht hinzugefügt sind können diese mit Add->TF->Ok eingebunden werden. TFs, wie z.B. forearm_link, wrist_1_link, lever, etc. sollten dann entsprechend ihrer tatsächlichen aktuellen Pose angezeigt werden.
```bash
$ rosrun rviz rviz
```


Soll nun die aktuelle Situation oder eine bestimmte Messfahrt (sobald mal das proceed mit y bestätigt) als rosbag aufgenommen werden kann dies mit folgenden Befehl ausgeführt werden. Ist die Messfahrt beedendet bzw. möchte man die Aufnahme beenden kann dies durch drücken von Strg+C in der gleichen Konsole erfolgen.

```bash
$ rosbag record -a
```

Es können mehrere Rosbags aufgenommen werden, diese erhalten als Namen jeweils den Zeitpunkt der Aufnahme, es lohnt sich jedoch immer die namen zu verändern, damit man die einzelnen Rosbags besser auseinanderhalten kann. Das Rosbag wird an der Stelle gespeichert, an der man den record Befehl von oben ausgeführt hat. Umbenennen geht in Linux mit dem mv (move) Befehl z.B.
```bash
$ mv 2019-11-19-19-56-09.bag mybag.bag
```


Um nun das Rosbag erneut abzuspielen lohnt es sich alle Prozesse, bis auf den roscore zu beenden (Strg+C) oder zu schließen.
Anschließend muss das Zeitsystem auf ein Simuliertes umgestellt werden. Zu beachten ist, dass wenn man nachher doch wieder eine neue 

```bash
$ rosparam set use_sim_time 1
(warum das?)
```

Abspielen der Rosbag
```bash
$ rosbag play thisisa.bag --clock --loop
```


## Reproduzieren der Daten als Rosbag

Wiederherstellen roboterfahrt (rviz) und Zeitbezug!
Problemstellung Benennen
Lösungsansatz vorstellen
```bash
$ vim createPtcl.py
```

T-Scan Punktwolke in Ros laden, intention

```bash
$ python createPtcl.py
```

Punktwolke an leverarm hängen (siehe py-skript)

Erzeugen einer Referenzierten Punktwolke

```bash
$ rosrun tf_points_global transform_point2pointcloud _ptcl2_global_frame:=map _ptcl2_local_frame:=lever _ptcl2_input_topic:=/tscan_cloud2 _ptcl2_output_topic:=/tscan_cloud2_global _drop_when_same_position:=false
```

Rausschreiben einer Referenzierten Punktwolke 
```bash
$ rosrun pcl_ros pointcloud_to_pcd input:=/tscan_cloud2_global
```

Zusammenfassen der pcds zu einer Ascii-Datei

```bash
$ printf '%s\n\n' "$(tail -n +30 *.pcd)" > ptcl_as_ascii.txt
```

Öffnen der Punktwolke mit CloudCompare zB.:

```bash
$ cloudcompare.CloudCompare ptcl_as_ascii.txt
```

Bekannte Probleme:
Ansätze:
Interpretationsansätze PW


---

### Problembeschreibung/Aufgabenstellung

1. Leverarm

2. Zeitsynchro

3. Datenverarbeitung
