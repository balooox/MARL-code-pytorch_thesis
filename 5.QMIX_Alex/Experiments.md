Mein nachgebauter Q-Mix Algorithmus konvergiert. Jedoch benötigt mein Algorithmus extrem lange. 

Mein Algorithmus:
lr = 4e-5
kein orthogonale Initalisierung der Gewichte
Zeitschritte bis Konvergenz: ~ 600.000

Originale Implementierung
lr = 5e-5
orthogonale Initalisierung der Gewichte
Zeitschritte bis Konvergenz: ~ 300.000

Unterschied in der Implementierung:
    Meine Implementierung: Für die Bestimmung des maximalen Q-Values des Target Network kommen ausschließlich die Q-Values der erlaubten Aktionen (im jeweiligen Zeitschritt) in Betracht

    Originalimplementierung: Zur Berechnung des Target Q-Values kommt zwar auch das Target Network zum Einsatz, jedoch wird die Aktion gewählt, die den höhsten Q-Value vom Eval Network bekommt. (erlaubte Aktionen mit betrachtet)

Frage:
Liegen die Unterschiede in der Konvergenzgeschwindigkeit an den unterschiedlichen Hyperparametern oder am Unterschied in der Implementierung?

Ideen:
- Orthogonale Initialisierung der Gewichte verwenden
- Orthogonale Initialisierung der Gewichte in der Originalimplementierung ausschalten
- Lernrate anpassen (statt 4e-5 -> 5e-5)
- Seed in der Originalimplementierung verändern 
