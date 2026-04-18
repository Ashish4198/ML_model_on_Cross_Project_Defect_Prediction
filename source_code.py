#!/usr/bin/env python3
"""
SAMPQIFS Cross Platform Defect Prediction - Optimized Version
Fixed: Faster optimization that completes properly
Author: AI Research Assistant
Date: August 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import time
import os

# Google Colab imports
try:
    from google.colab import files
    COLAB_ENV = True
    print("✅ Google Colab environment detected")
except ImportError:
    COLAB_ENV = False
    print("⚠️  Not in Google Colab - using local file input")

warnings.filterwarnings('ignore')

class QuantumChromosome:
    """Simplified quantum chromosome"""

    def __init__(self, num_features):
        self.num_features = num_features
        self.probabilities = np.random.uniform(0.3, 0.7, num_features)
        self.fitness = 0.0

    def measure(self):
        """Get binary solution"""
        solution = (np.random.rand(self.num_features) < self.probabilities).astype(int)
        if np.sum(solution) < 2:
            indices = np.random.choice(self.num_features, 2, replace=False)
            solution[indices] = 1
        return solution

    def update(self, best_solution, rate=0.1):
        """Update probabilities"""
        for i in range(self.num_features):
            if best_solution[i] == 1:
                self.probabilities[i] += rate * (1 - self.probabilities[i])
            else:
                self.probabilities[i] -= rate * self.probabilities[i]
        self.probabilities = np.clip(self.probabilities, 0.1, 0.9)

class FastSAMPQIFS:
    """Optimized SAMPQIFS for faster execution"""

    def __init__(self):
        # Reduced parameters for faster execution
        self.num_populations = 3
        self.population_size = 8
        self.max_iterations = 10  # Reduced from 15
        self.populations = []
        self.best_solution = None
        self.best_fitness = 0.0

        # Fast classifier for evaluation
        self.eval_classifier = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)

    def evaluate_fitness(self, chromosome, X_train, y_train):
        """Fast fitness evaluation"""
        solution = chromosome.measure()
        selected = np.where(solution == 1)[0]

        if len(selected) == 0:
            return 0.0

        X_selected = X_train[:, selected]

        try:
            # Single train-test split for speed (instead of cross-validation)
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_selected, y_train, test_size=0.3, random_state=42, stratify=y_train
            )

            self.eval_classifier.fit(X_tr, y_tr)
            y_pred_proba = self.eval_classifier.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)

            # Simple fitness with feature penalty
            feature_penalty = len(selected) / len(solution)
            fitness = auc_score * (1 - 0.1 * feature_penalty)

        except Exception:
            fitness = 0.0

        chromosome.fitness = fitness
        return fitness

    def optimize(self, X_train, y_train):
        """Fast optimization"""
        num_features = X_train.shape[1]

        print(f"🧬 Fast SAMPQIFS Started")
        print(f"   Features: {num_features}")
        print(f"   Populations: {self.num_populations} x {self.population_size}")
        print(f"   Iterations: {self.max_iterations}")

        # Initialize populations
        self.populations = []
        for pop_id in range(self.num_populations):
            population = []
            for _ in range(self.population_size):
                chromosome = QuantumChromosome(num_features)
                # Different initialization strategies
                if pop_id == 0:
                    chromosome.probabilities = np.random.uniform(0.2, 0.4, num_features)
                elif pop_id == 1:
                    chromosome.probabilities = np.random.uniform(0.6, 0.8, num_features)
                else:
                    chromosome.probabilities = np.random.uniform(0.3, 0.7, num_features)
                population.append(chromosome)
            self.populations.append(population)

        # Evolution loop
        for generation in range(self.max_iterations):
            print(f"   Gen {generation+1}/{self.max_iterations}...", end=" ", flush=True)

            # Evaluate all populations
            for pop_idx, population in enumerate(self.populations):
                for chromosome in population:
                    self.evaluate_fitness(chromosome, X_train, y_train)

                # Sort and update best
                population.sort(key=lambda x: x.fitness, reverse=True)
                if population[0].fitness > self.best_fitness:
                    self.best_fitness = population[0].fitness
                    self.best_solution = population[0].measure().copy()

            # Create new generation
            for pop_idx, population in enumerate(self.populations):
                new_pop = []
                # Keep best 2
                new_pop.extend(population[:2])

                # Generate offspring
                while len(new_pop) < self.population_size:
                    parent1 = population[np.random.randint(0, 3)]  # Top 3
                    parent2 = population[np.random.randint(0, 3)]

                    child = QuantumChromosome(num_features)
                    child.probabilities = (parent1.probabilities + parent2.probabilities) / 2

                    # Mutation
                    if np.random.random() < 0.1:
                        mask = np.random.random(num_features) < 0.1
                        child.probabilities[mask] = np.random.uniform(0.2, 0.8, np.sum(mask))

                    # Update based on best
                    if self.best_solution is not None:
                        child.update(self.best_solution, 0.1)

                    new_pop.append(child)

                self.populations[pop_idx] = new_pop

            # Progress
            selected_count = np.sum(self.best_solution) if self.best_solution is not None else 0
            print(f"Fitness={self.best_fitness:.4f}, Features={selected_count}/{num_features}")

        print(f"✅ SAMPQIFS Complete: {np.sum(self.best_solution)}/{num_features} features selected")
        return self.best_solution

class SAMPQIFSPredictor:
    """Main prediction class"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.sampqifs = FastSAMPQIFS()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.selected_features = None
        self.feature_names = None
        self.is_trained = False

        print("🔬 SAMPQIFS Defect Predictor Ready")
        print("   Algorithm: Fast Self-Adaptive Multi-Population Quantum-Inspired Feature Selection")
        print("   Classifier: Random Forest")
        print("   Metric: AUC")

    def parse_arff(self, file_content, filename="file.arff"):
        """Parse ARFF file"""
        try:
            if isinstance(file_content, bytes):
                content = file_content.decode('utf-8')
            else:
                content = file_content
            lines = content.strip().split('\n')
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            return None, None, None

        data_section = False
        features = []
        feature_names = []
        labels = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            if line.lower().startswith('@attribute'):
                parts = line.split()
                if len(parts) >= 2:
                    feature_names.append(parts[1].strip('\'"'))

            elif line.lower().startswith('@data'):
                data_section = True
                continue

            elif data_section:
                try:
                    values = [v.strip().strip('\'"') for v in line.split(',')]
                    if len(values) >= 2:
                        feature_row = []
                        for val in values[:-1]:
                            try:
                                feature_row.append(float(val))
                            except ValueError:
                                feature_row.append(abs(hash(val)) % 1000)
                        features.append(feature_row)
                        labels.append(values[-1].lower())
                except:
                    continue

        if not features:
            print(f"❌ No data in {filename}")
            return None, None, None

        X = np.array(features, dtype=float)
        le = LabelEncoder()
        y = le.fit_transform(labels)

        if feature_names:
            feature_names = feature_names[:-1]
        else:
            feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]

        print(f"📁 {filename}: {X.shape[0]} samples, {X.shape[1]} features, {np.mean(y):.1%} defects")
        return X, y, feature_names

    def train(self, file_content, filename="training.arff"):
        """Train with SAMPQIFS"""
        print("\n" + "="*60)
        print("🎯 SAMPQIFS TRAINING")
        print("="*60)

        X_train, y_train, self.feature_names = self.parse_arff(file_content, filename)
        if X_train is None:
            return False

        print("\n🔄 Preprocessing...")
        X_scaled = self.scaler.fit_transform(X_train)

        print("\n🧬 Running SAMPQIFS...")
        start_time = time.time()
        self.selected_features = self.sampqifs.optimize(X_scaled, y_train)
        optimization_time = time.time() - start_time

        X_selected = X_scaled[:, self.selected_features.astype(bool)]

        selected_names = [self.feature_names[i] for i in range(len(self.feature_names))
                         if self.selected_features[i]]

        print(f"\n📊 Selected Features ({len(selected_names)}):")
        for i, name in enumerate(selected_names[:8], 1):
            print(f"   {i}. {name}")
        if len(selected_names) > 8:
            print(f"   ... and {len(selected_names) - 8} more")

        print(f"\n🤖 Training Random Forest...")
        self.classifier.fit(X_selected, y_train)

        # Quick validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.classifier, X_selected, y_train,
                                   cv=cv, scoring='roc_auc', n_jobs=-1)

        print(f"\n📈 Training Results:")
        print(f"   Cross-Validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"   Features: {len(selected_names)}/{len(self.feature_names)} ({(1-len(selected_names)/len(self.feature_names)):.1%} reduction)")
        print(f"   Optimization Time: {optimization_time:.1f} seconds")

        self.is_trained = True
        return True

    def test(self, file_content, dataset_name="Test", filename="test.arff"):
        """Test the model"""
        if not self.is_trained:
            print("❌ Train the model first!")
            return None

        print(f"\n📊 Testing on {dataset_name}...")

        X_test, y_test, _ = self.parse_arff(file_content, filename)
        if X_test is None:
            return None

        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = X_test_scaled[:, self.selected_features.astype(bool)]

        y_pred = self.classifier.predict(X_test_selected)
        y_pred_proba = self.classifier.predict_proba(X_test_selected)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
        else:
            accuracy = np.mean(y_pred == y_test)
            precision = recall = f1 = 0.0

        results = {
            'dataset': dataset_name,
            'samples': len(y_test),
            'defect_rate': np.mean(y_test),
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print(f"   📊 Results: {len(y_test)} samples, {np.mean(y_test):.1%} defects")
        print(f"   🎯 AUC: {auc:.4f}")
        print(f"   📈 Accuracy: {accuracy:.4f}")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   📊 Recall: {recall:.4f}")
        print(f"   🔗 F1-Score: {f1:.4f}")

        return results

def upload_file(prompt="📂 Upload ARFF file"):
    """File upload function"""
    if COLAB_ENV:
        print(f"{prompt}")
        uploaded = files.upload()
        if not uploaded:
            return None, None
        filename = list(uploaded.keys())[0]
        return uploaded[filename], filename
    else:
        filepath = input(f"{prompt} - Enter path: ").strip()
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None, None
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read(), os.path.basename(filepath)

def main():
    """Main function"""
    print("🚀 SAMPQIFS Cross Platform Defect Prediction")
    print("="*60)
    print("🧬 Self-Adaptive Multi-Population Quantum-Inspired Feature Selection")
    print("🎯 Primary Metric: AUC")

    predictor = SAMPQIFSPredictor()

    # TRAINING
    print(f"\n🎯 TRAINING PHASE")
    train_content, train_filename = upload_file("📂 Upload Training ARFF File")

    if train_content is None:
        print("❌ No training file!")
        return

    if not predictor.train(train_content, train_filename):
        print("❌ Training failed!")
        return

    # TESTING
    print("\n" + "="*60)
    print("🧪 TESTING PHASE")
    print("="*60)
    print("Now upload test datasets from different projects...")

    all_results = []
    test_count = 1

    while True:
        print(f"\n📊 Test Dataset #{test_count}")
        test_content, test_filename = upload_file(f"📂 Upload Test ARFF File #{test_count}")

        if test_content is None:
            if test_count == 1:
                print("❌ Need at least one test file!")
                continue
            else:
                print("✅ Testing complete!")
                break

        dataset_name = input(f"Dataset name (or Enter for '{test_filename}'): ").strip()
        if not dataset_name:
            dataset_name = test_filename.replace('.arff', '')

        results = predictor.test(test_content, dataset_name, test_filename)
        if results:
            all_results.append(results)
            test_count += 1

        if not COLAB_ENV:
            more = input("\nTest another dataset? (y/n): ").lower().strip()
            if more != 'y':
                break

    # SUMMARY
    if all_results:
        print("\n" + "="*60)
        print("📊 CROSS PLATFORM RESULTS SUMMARY")
        print("="*60)

        if len(all_results) > 1:
            print(f"{'Dataset':<15} {'AUC':<8} {'Accuracy':<9} {'Precision':<9} {'Recall':<8} {'F1':<8}")
            print("-" * 60)

            for result in all_results:
                print(f"{result['dataset']:<15} "
                      f"{result['auc']:<8.4f} "
                      f"{result['accuracy']:<9.4f} "
                      f"{result['precision']:<9.4f} "
                      f"{result['recall']:<8.4f} "
                      f"{result['f1_score']:<8.4f}")

            auc_scores = [r['auc'] for r in all_results]
            print("-" * 60)
            print(f"{'AVERAGE':<15} {np.mean(auc_scores):<8.4f}")
            print(f"\n🏆 Best AUC: {max(auc_scores):.4f}")
            print(f"📊 Average AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

        else:
            result = all_results[0]
            print(f"✅ Single test complete: AUC = {result['auc']:.4f}")

        # Feature importance
        if hasattr(predictor.classifier, 'feature_importances_'):
            print(f"\n🏆 Top 5 Important Features:")
            importances = predictor.classifier.feature_importances_
            selected_names = [predictor.feature_names[i] for i in range(len(predictor.feature_names))
                            if predictor.selected_features[i]]

            pairs = list(zip(selected_names, importances))
            pairs.sort(key=lambda x: x[1], reverse=True)

            for i, (feature, importance) in enumerate(pairs[:5], 1):
                print(f"   {i}. {feature}: {importance:.4f}")

    print(f"\n✅ SAMPQIFS Cross Platform Defect Prediction Complete!")
    if predictor.selected_features is not None:
        selected = np.sum(predictor.selected_features)
        total = len(predictor.feature_names)
        print(f"📊 Final: {selected}/{total} features selected, {len(all_results)} datasets tested")

if __name__ == "__main__":
    main()
