# Proiect: Recunoașterea atacului cerebral și gestionarea programărilor pacienților

## Descriere
Acest proiect a fost dezvoltat pentru a ajuta spitalele în:
- **Recunoașterea unei persoane care suferă un atac cerebral**, utilizând algoritmi de procesare a imaginilor și inteligență artificială.
- **Gestionarea pacienților**, prin recunoașterea acestora și afișarea sălii în care sunt programați.

Proiectul a fost dezvoltat și testat pe un **Raspberry Pi cu 4 GB RAM**, fiind optimizat pentru resurse hardware limitate.  
Acesta a fost realizat în cadrul unui **hackathon**, alături de echipa **A++ și Aditia**.

## Funcționalități principale
1. **Recunoașterea atacului cerebral**:
   - Detectarea semnelor unui atac cerebral în timp real folosind camere video și algoritmi AI.
   - Alertarea imediată a personalului medical în caz de urgență.

2. **Gestionarea pacienților**:
   - Identificarea pacienților pe baza datelor biometrice (de exemplu, recunoaștere facială).
   - Afișarea sălii de consultație sau tratament conform programării.

## Tehnologii utilizate
- **Python**: Limbajul principal de programare.
- **OpenCV**: Pentru procesarea imaginilor.
- **Machine Learning/Deep Learning**: Pentru recunoașterea semnelor de atac cerebral.
- **SQLite**: Pentru gestionarea bazei de date a pacienților și programărilor.
- **Interfață utilizator**: Dezvoltată pentru afișarea rapidă a informațiilor despre pacienți și programări.

## Cerințe de sistem
- **Raspberry Pi** cu 4 GB RAM (sau echivalentul său).
- Cameră video pentru capturarea imaginilor pacienților.
- Conexiune la rețea pentru a trimite alerte și a accesa baza de date.

## Cum funcționează
1. Sistemul monitorizează pacienții printr-o rețea de camere conectate.
2. Dacă un pacient prezintă semne de atac cerebral, algoritmii AI detectează acest lucru și trimit o alertă personalului medical.
3. Datele despre pacienți sunt stocate într-o bază de date și afișate pe baza programărilor lor.

## Instalare și utilizare
1. Clonează acest repository:
   ```bash
   git clone https://github.com/iAndreea02/A-_Aditia.git
