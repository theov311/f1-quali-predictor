import fastf1
import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta

# Configuration de l'affichage et du cache
pd.set_option('display.float_format', '{:.3f}'.format)
# REMARQUE : Créez un dossier 'cache' dans le répertoire du script ou désactivez le cache
try:
    fastf1.Cache.enable_cache('cache') 
except:
    print("Avertissement: Cache non activé. Le chargement des données sera plus lent.")

class QualiPredictor:
    def __init__(self, year, gp, session_type='Q'):
        """
        Initialise la session et charge les données.
        """
        print(f"Chargement de la session : {year} {gp} ({session_type})...")
        self.session = fastf1.get_session(year, gp, session_type)
        self.session.load(telemetry=False, weather=False, messages=False)
        self.laps = self.session.laps
        print("Données chargées avec succès.")
        
        # Paramètres de simulation
        self.evolution_rate = 0.03  # Secondes gagnées par minute (moyenne)
        
    def _filter_laps_at_time(self, current_time_minutes):
        """
        Simule le temps réel en ne retournant que les tours complétés 
        avant un certain moment de la session.
        """
        # Convertir les minutes en Timedelta
        current_time = timedelta(minutes=current_time_minutes)
        
        # Filtre 1 : Tours complétés avant le temps actuel
        # La colonne 'Time' dans FastF1 est le temps de session à la fin du tour
        known_laps = self.laps[self.laps['Time'] <= current_time]
        
        # Filtre 2 : Tours valides (Pas de tour de sortie, pas de drapeau jaune majeur sur le tour)
        # IsAccurate = True garantit que les secteurs et le temps total sont cohérents
        # On les applique séquentiellement et on gère les erreurs
        try:
            clean_laps = known_laps.pick_accurate()
        except:
            clean_laps = known_laps
            
        try:
            clean_laps = clean_laps.pick_wo_box()
        except:
            pass  # Si pick_wo_box échoue, on garde les laps de toute façon
        
        # Si aucun tour n'a été trouvé après filtrage strict, on retourne tous les tours du temps donné
        if clean_laps.empty:
            clean_laps = known_laps
        
        return clean_laps

    def _calculate_prediction_interval(self, lap_times, confidence=0.90):
        """
        Calcule un intervalle de prédiction basé sur la distribution t-Student.
        Retourne (lower_bound, predicted_mean, upper_bound)
        """
        n = len(lap_times)
        if n < 2:
            return None # Pas assez de données pour une statistique fiable
            
        mean = np.mean(lap_times)
        
        # Si n == 2, on utilise une estimation simple
        if n == 2:
            margin = abs(lap_times.iloc[1] - lap_times.iloc[0]) * 0.5
            return (mean - margin, mean, mean + margin)
        
        sem = stats.sem(lap_times) # Erreur standard
        
        # Calcul de l'intervalle
        # On utilise t.ppf pour obtenir la valeur critique de t
        t_crit = stats.t.ppf((1 + confidence) / 2., n-1)
        margin = t_crit * sem
        
        return (mean - margin, mean, mean + margin)

    def predict_q1_cutoff(self, simulation_time_min, time_remaining_min):
        """
        Prédit le temps pour P15 (Cutoff Q1).
        """
        current_data = self._filter_laps_at_time(simulation_time_min)
        
        # On prend le meilleur tour de chaque pilote jusqu'à présent
        # Filtrer les NaN dans LapTime
        valid_laps = current_data[current_data['LapTime'].notna()]
        
        best_laps = valid_laps.groupby('Driver')['LapTime'].min().sort_values()
        
        # Convertir en secondes pour les calculs
        best_laps_sec = best_laps.dt.total_seconds()
        
        if len(best_laps) < 3:
            return f"Données insuffisantes ({len(best_laps)} pilotes avec tours valides, attendre plus de tours)"
            
        # LA BULLE : On regarde les pilotes classés entre P13 et P17
        # Ce sont eux qui définissent la limite, pas Max Verstappen en P1.
        # On ajuste les indices en fonction du nombre de pilotes disponibles
        bubble_start = min(1, len(best_laps) - 1) if len(best_laps) > 2 else 0
        bubble_end = min(len(best_laps), max(3, len(best_laps) // 2))
        bubble_times = best_laps_sec.iloc[bubble_start:bubble_end]
        
        if len(bubble_times) < 2:
            return "Pas assez de pilotes dans la zone de danger pour faire une prédiction"
        
        stats_result = self._calculate_prediction_interval(bubble_times)
        
        if stats_result:
            low, mean, high = stats_result
            # Application du facteur d'évolution de piste
            # Les temps vont baisser d'ici la fin de séance
            evo_gain = self.evolution_rate * time_remaining_min
            
            pred_low = low - evo_gain
            pred_high = high - evo_gain
            
            # Affichage du cutoff (normalement P15)
            cutoff_position = min(15, len(best_laps))
            cutoff_time = best_laps_sec.iloc[cutoff_position - 1] if cutoff_position <= len(best_laps) else None
            
            return {
                "Cible": f"P{cutoff_position} (Q2 Cutoff)",
                "Temps Actuel Cutoff": cutoff_time,
                "Meilleur temps": best_laps_sec.iloc[0],
                "Intervalle Prédit": (pred_low, pred_high),
                "Pilotes en danger": list(best_laps.index[bubble_start:bubble_end])
            }
        return "Calcul impossible"

    def predict_q3_pole(self, simulation_time_min, time_remaining_min):
        """
        Prédit le temps de la Pole Position (P1) en Q3.
        Utilise la méthode 'Ultimate Lap' (meilleurs secteurs combinés).
        """
        # En Q3, le temps de simulation est décalé (Q3 commence après Q1+Q2)
        # Pour simplifier la démo, on considère simulation_time_min comme temps absolu depuis le début de Q3
        # Il faut charger uniquement les données Q3 ou filtrer par temps absolu de session
        
        # Note: FastF1 SessionTime est continu. Q3 commence généralement après ~45-50 min.
        # Ici, on filtre simplement sur les tours rapides disponibles à l'instant T.
        current_data = self._filter_laps_at_time(simulation_time_min)
        
        if current_data.empty:
            return "Pas de données Q3"

        # 1. Meilleur temps de tour (Current Pole)
        try:
            best_lap_times = current_data['LapTime'].dropna()
            if best_lap_times.empty:
                return "Pas de temps de tour disponible"
            
            current_pole = best_lap_times.min().total_seconds()
            
            # 2. Theoretical Best Lap (Ultimate) de la session
            # Utilise les meilleurs secteurs disponibles
            try:
                best_s1 = current_data['Sector1Time'].dropna().min().total_seconds()
                best_s2 = current_data['Sector2Time'].dropna().min().total_seconds()
                best_s3 = current_data['Sector3Time'].dropna().min().total_seconds()
                ultimate_lap = best_s1 + best_s2 + best_s3
            except:
                # Si les données sectorielles manquent, on utilise une estimation
                ultimate_lap = current_pole * 0.98  # Environ 2% plus rapide que la pole actuelle
            
        except:
            return "Données insuffisantes"

        # 3. Prédiction
        # La pole sera entre l'Ultimate Lap (perfection théorique) et la Pole Actuelle
        # On applique aussi l'évolution de piste
        evo_gain = self.evolution_rate * time_remaining_min
        
        # La borne basse est l'ultimate lap moins l'évolution (très agressif)
        # La borne haute est la pole actuelle moins l'évolution
        pred_low = ultimate_lap - (evo_gain * 0.5) # On applique moins d'evo sur l'ultimate car déjà très rapide
        pred_high = current_pole - evo_gain
        
        return {
            "Cible": "P1 (Pole Position)",
            "Pole Provisoire": current_pole,
            "Ultimate Lap Théorique": ultimate_lap,
            "Intervalle Prédit": (pred_low, pred_high)
        }

# ==========================================
# EXEMPLE D'EXÉCUTION (Simulation)
# ==========================================

if __name__ == "__main__":
    # Exemple : Qualifications Monaco 2023
    predictor = QualiPredictor(2023, 'Monaco', 'Q')
    
    print("\n--- SIMULATION Q1 : Il reste 5 minutes ---")
    # Q1 dure 18 min. À t=15min (reste 3), quelle est la projection pour P15?
    # Les premiers tours commencent à ~14:24, donc 15 min est un bon point de contrôle
    q1_result = predictor.predict_q1_cutoff(simulation_time_min=15, time_remaining_min=3)
    print(q1_result)
    
    print("\n--- SIMULATION Q3 : Il reste 2 minutes (Money Time) ---")
    # Q3 commence tard dans la session. Disons à t=70min (proche de la fin).
    q3_result = predictor.predict_q3_pole(simulation_time_min=70, time_remaining_min=2)
    print(q3_result)