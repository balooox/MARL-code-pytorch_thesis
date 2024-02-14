In der Datei Experiments.md bin auf den einzigen wirklichen Implementierungsunterschied zwischen meiner und der Originalimplementation eingegangen. An dieser Stelle möchte ich noch mehr Input liefern:

# Unterschiede in der Implementierung #

Bei dem Agentennetzwerk handelt sich nicht um eine Q-Funktion, sondern um eine Utility-Funktion. (siehe Anmerkung im Paper). Der Einfachkeit halber wird diese jedoch auch als Q-Funktion betitelt. Bei der Bestimmung des Target-Wertes muss daher auch nicht wie bei Q-Learning sonst üblich der beste Wert des nächsten States verwendet werden (arg max a Q(s', a)). Im Fall von Q-Mix bestimmt das Target-Network den Utility-Value der bereits von Eval-Q-Network verwendeten Aktion neu. Es findet also eine Neubewertung der Aktion statt.

Die eigentliche Bestimmunng des Q-Target-Values findet erst nach der Berechung des Q_tot-Wertes statt. An dieser Stelle wird die bekannt Formel verwendet.

Q_tot = state_reward + gamma * Q_tot_target,    wobei Q_tot_target = Target_Mix(next_state)

Bei meiner Implementierung wird für die Bestimmung des Target-Wertes der maximale Wert vom Q-Target-Network zurückgegeben (beste Aktion im nächsten State). Das Target-NN entscheidet also über die beste Aktion, nimmt nicht die bereits verwendetet Funktion.

Trotz des Unterschieds in der Implementierung konvergiert meine Version von Q-Mix

Offene Frage: Was sagt die Originalimplementierung (pymarl) ?  
