## Unterschiedliche Konvergenzzeiten ##

Mein nachgebauter Q-Mix Algorithmus konvergiert. Jedoch benötigt mein Algorithmus extrem lange. 

Mein Algorithmus:
lr = 4e-5
kein orthogonale Initalisierung der Gewichte
Zeitschritte bis Konvergenz: ~ 600.000

Originale Implementierung
lr = 5e-4
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

Antwort:
Meine Lernrate lag bei 4e-5, während die Lernrate de Originalimplementierung bei 5e-4 lag. Damit lag die Lernrate beim 12,5-fachen. Hier einfach schlusslig gewesen. Unterschied in der 10ner-Potenz ist mir nicht aufgefallen. Nach der Änderung hat mein Algorithmus Konvergenz nach 200.000 Zeitschritten erreicht. Keine Veränderung am Code vorgenommen. Andere Unterschiede bleiben bestehen. 


## Weitere Infos ##
Auch unter einem veränderten Seed konvergiert der Originalalgorithmus. Nicht seedgebunden. Damit Algemeingültigkeit bestätigt.

Im Abschnitt 'Unterschiede in den Konvergenzzeiten' bin auf den einzigen wirklichen Implementierungsunterschied eingegangen. An dieser Stelle möchte ich noch mehr Input liefern:

# Unterschiede in der Implementierung #

Bei dem Agentennetzwerk handelt sich nicht um eine Q-Funktion, sondern um eine Utility-Funktion. (siehe Anmerkung im Paper). Der Einfachkeit halber wird diese jedoch auch als Q-Funktion betitelt. Bei der Bestimmung des Target-Wertes muss daher auch nicht wie bei Q-Learning sonst üblich der beste Wert des nächsten States verwendet werden (arg max a Q(s', a)). Im Fall von Q-Mix bestimmt das Target-Network den Utility-Value der bereits von Eval-Q-Network verwendeten Aktion neu. Es findet also eine Neubewertung der Aktion statt.

Die eigentliche Bestimmunng des Q-Target-Values findet erst nach der Berechung des Q_tot-Wertes statt. An dieser Stelle wird die bekannt Formel verwendet.

Q_tot = state_reward + gamma * Q_tot_target,    wobei Q_tot_target = Target_Mix(next_state)

Bei meiner Implementierung wird für die Bestimmung des Target-Wertes der maximale Wert vom Q-Target-Network zurückgegeben (beste Aktion im nächsten State). Das Target-NN entscheidet also über die beste Aktion, nimmt nicht die bereits verwendetet Funktion.

Trotz des Unterschieds in der Implementierung konvergiert meine Version von Q-Mix

Offene Frage: Was sagt die Originalimplementierung (pymarl) ?  

