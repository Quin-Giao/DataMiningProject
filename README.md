# Preprocessing Tools

## Csv2Arff.java

Converts CSV files to ARFF format.

**Compilation:**
```bash
javac -cp weka.jar Csv2Arff.java
```

**Usage:**
```bash
java -cp .:weka.jar Csv2Arff heart_disease.csv
java -cp .:weka.jar Csv2Arff heart_disease.csv heart_disease_cleaned.arff
```

---

## LoadData.java

Loads and displays ARFF file information.

**Compilation:**
```bash
javac -cp weka.jar LoadData.java
```

**Usage:**
```bash
java -cp .:weka.jar LoadData heart_disease_cleaned.arff
```

**Output:**
```
Relation: heart_disease
Instances: 303
Attributes: 14
Class attribute: target
```
