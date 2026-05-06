# Σύγκριση Αλγορίθμων Βελτιστοποίησης της Steepest Descent

Αυτό το project υλοποιεί, συγκρίνει και οπτικοποιεί διάφορες μεθόδους βελτιστοποίησης πρώτης τάξης πάνω σε γνωστές συναρτήσεις ελέγχου, τόσο κυρτές όσο και μη-κυρτές.
Η κύρια αναφορά είναι στο αρχείο [`steepest_descent_optimization.pdf`](steepest_descent_optimization.pdf).
Επίσης, όλες οι προσωπικές μου σημειώσεις για την θεωρία που χρησιμοποιήθηκε βρίσκονται στο αρχείο [`personal-notes.pdf`](personal-notes.pdf).

## Αλγόριθμοι που Υλοποιήθηκαν
- **Steepest Descent**.
- **Heavy Ball**.
- **Nesterov Accelerated Gradient (NAG)**.
- **Barzilai-Borwein (BB1)**.

## Συναρτήσεις Ελέγχου
- **Κυρτές/Convex:** Sphere, Quadratic, Booth, Matyas.
- **Μη-Κυρτές/Non-Convex:** Rastrigin, Rosenbrock, Ackley.

## Δομή του Project
Το project χωρίζεται σε 4 βασικά αρχεία Python:

* `functions.py`: Μαθηματικοί τύποι και κλίσεις των συναρτήσεων ελέγχου.
* `methods.py`: Υλοποίηση των αλγορίθμων βελτιστοποίησης.
* `comparison.py`: Εκτελεί στατιστικά πειράματα για όλες τις μεθόδους και τις συναρτήσεις. Υπολογίζει τη μέση τιμή και την τυπική απόκλιση για το τελικό σφάλμα και τον αριθμό επαναλήψεων.
* `visualize.py`: Δημιουργεί animations για δύο παραδείγματα.

## Requirements
```bash
pip install numpy scipy matplotlib