from flask import Flask, render_template, request, jsonify
import fastf1
import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta
import traceback

# Configuration de l'affichage et du cache
pd.set_option('display.float_format', '{:.3f}'.format)
try:
    fastf1.Cache.enable_cache('cache') 
except:
    print("Avertissement: Cache non activé. Le chargement des données sera plus lent.")

app = Flask(__name__)

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
        
        # Détecter les plages de temps pour Q1, Q2, Q3
        self._detect_session_ranges()
    
    def _detect_session_ranges(self):
        """
        Détecte automatiquement les plages de temps pour Q1, Q2, Q3 dans les données.
        FastF1 fournit un temps continu pour toute la session.
        """
        if len(self.laps) == 0:
            self.session_ranges = {'Q1': None, 'Q2': None, 'Q3': None}
            return
        
        # Trier les tours par temps
        sorted_laps = self.laps.sort_values('Time')
        times_minutes = sorted_laps['Time'].dt.total_seconds() / 60
        
        # Trouver les gaps importants (plus de 5 minutes) qui séparent Q1, Q2, Q3
        time_diffs = times_minutes.diff()
        gap_indices = time_diffs[time_diffs > 5].index
        
        # Diviser en sessions
        if len(gap_indices) >= 2:
            # On a les 3 sessions distinctes
            q1_start = times_minutes.min()
            q1_end = times_minutes.loc[gap_indices[0]]
            q2_start = times_minutes.loc[gap_indices[0]]
            q2_end = times_minutes.loc[gap_indices[1]]
            q3_start = times_minutes.loc[gap_indices[1]]
            q3_end = times_minutes.max()
        elif len(gap_indices) == 1:
            # Seulement 2 sessions distinctes
            q1_start = times_minutes.min()
            q1_end = times_minutes.loc[gap_indices[0]]
            q2_start = times_minutes.loc[gap_indices[0]]
            q2_end = times_minutes.max()
            q3_start = q2_end
            q3_end = q2_end
        else:
            # Tout est continu, diviser approximativement
            total_duration = times_minutes.max() - times_minutes.min()
            q1_start = times_minutes.min()
            q1_end = q1_start + (total_duration * 0.4)
            q2_start = q1_end
            q2_end = q2_start + (total_duration * 0.35)
            q3_start = q2_end
            q3_end = times_minutes.max()
        
        self.session_ranges = {
            'Q1': (q1_start, q1_end),
            'Q2': (q2_start, q2_end),
            'Q3': (q3_start, q3_end)
        }
        
        print(f"DEBUG: Session ranges detected:")
        print(f"  Q1: {q1_start:.1f} - {q1_end:.1f} min")
        print(f"  Q2: {q2_start:.1f} - {q2_end:.1f} min")
        print(f"  Q3: {q3_start:.1f} - {q3_end:.1f} min")
        
    def _filter_laps_at_time(self, current_time_minutes):
        """
        Simule le temps réel en ne retournant que les tours complétés 
        avant un certain moment de la session.
        """
        # Convertir les minutes en Timedelta
        current_time = timedelta(minutes=current_time_minutes)
        
        # Filtre 1 : Tours complétés avant le temps actuel
        known_laps = self.laps[self.laps['Time'] <= current_time]
        
        print(f"DEBUG: Tours trouvés avant {current_time_minutes} min: {len(known_laps)}")
        
        # Retourner directement tous les tours sans filtrage strict
        # Le filtrage des tours valides se fera dans predict_q1_cutoff
        return known_laps

    def _calculate_prediction_interval(self, lap_times, confidence=0.90):
        """
        Calcule un intervalle de prédiction basé sur la distribution t-Student.
        Retourne (lower_bound, predicted_mean, upper_bound)
        """
        n = len(lap_times)
        if n < 2:
            return None
            
        mean = np.mean(lap_times)
        
        if n == 2:
            margin = abs(lap_times.iloc[1] - lap_times.iloc[0]) * 0.5
            return (mean - margin, mean, mean + margin)
        
        sem = stats.sem(lap_times)
        t_crit = stats.t.ppf((1 + confidence) / 2., n-1)
        margin = t_crit * sem
        
        return (mean - margin, mean, mean + margin)

    def predict_q1_cutoff(self, simulation_time_min, time_remaining_min):
        """
        Prédit le temps pour P15 (Cutoff Q1).
        """
        current_data = self._filter_laps_at_time(simulation_time_min)
        
        print(f"DEBUG Q1: Tours totaux: {len(current_data)}")
        
        # Filtrer seulement les tours avec un temps valide (pas de NaN)
        valid_laps = current_data[(current_data['LapTime'].notna()) & (current_data['LapTime'] > timedelta(0))]
        
        print(f"DEBUG Q1: Tours avec LapTime valide: {len(valid_laps)}")
        
        if len(valid_laps) == 0:
            return {
                "success": False,
                "error": f"Aucun tour valide trouvé à {simulation_time_min} minutes. Essayez un moment plus tardif dans la session (ex: après 15-16 minutes)."
            }
        
        # Obtenir le meilleur tour de chaque pilote
        best_laps = valid_laps.groupby('Driver')['LapTime'].min().sort_values()
        
        print(f"DEBUG Q1: Pilotes avec tours: {len(best_laps)}, Pilotes: {list(best_laps.index)}")
        
        # Convertir en secondes pour les calculs
        best_laps_sec = best_laps.dt.total_seconds()
        
        # Accepter au moins 2 pilotes au lieu de 3 pour plus de flexibilité
        if len(best_laps) < 2:
            return {
                "success": False,
                "error": f"Seulement {len(best_laps)} pilote(s) avec des tours valides. Essayez un moment plus tardif (recommandé: après 16 minutes)."
            }
            
        # LA BULLE : On regarde les pilotes dans la zone médiane/basse
        # Si peu de pilotes, on prend tous ceux disponibles
        if len(best_laps) <= 5:
            bubble_times = best_laps_sec  # Prendre tous les pilotes
        else:
            # Prendre les pilotes du milieu vers le bas
            bubble_start = len(best_laps) // 3  # Commencer au tiers
            bubble_end = min(len(best_laps), bubble_start + 5)  # Prendre 5 pilotes
            bubble_times = best_laps_sec.iloc[bubble_start:bubble_end]
        
        print(f"DEBUG Q1: Bubble de {len(bubble_times)} pilotes")
        
        if len(bubble_times) < 2:
            return {
                "success": False,
                "error": f"Pas assez de pilotes dans la zone de prédiction. Essayez un moment plus tardif."
            }
        
        stats_result = self._calculate_prediction_interval(bubble_times)
        
        if stats_result:
            low, mean, high = stats_result
            evo_gain = self.evolution_rate * time_remaining_min
            
            pred_low = low - evo_gain
            pred_high = high - evo_gain
            
            cutoff_position = min(15, len(best_laps))
            cutoff_time = best_laps_sec.iloc[cutoff_position - 1] if cutoff_position <= len(best_laps) else None
            
            return {
                "success": True,
                "Cible": f"P{cutoff_position} (Q2 Cutoff)",
                "Temps Actuel Cutoff": float(cutoff_time) if cutoff_time else None,
                "Meilleur temps": float(best_laps_sec.iloc[0]),
                "Intervalle Prédit": (float(pred_low), float(pred_high)),
                "Pilotes en danger": list(best_laps.index[bubble_start:bubble_end])
            }
        return None

    def predict_q3_pole(self, simulation_time_min, time_remaining_min):
        """
        Prédit le temps de la Pole Position (P1) en Q3.
        """
        current_data = self._filter_laps_at_time(simulation_time_min)
        
        if current_data.empty:
            return {
                "success": False,
                "error": f"Aucun tour trouvé à {simulation_time_min} minutes. Essayez un moment plus tardif dans la session."
            }

        try:
            best_lap_times = current_data['LapTime'].dropna()
            if best_lap_times.empty:
                return {
                    "success": False,
                    "error": f"Aucun temps de tour valide à {simulation_time_min} minutes. Essayez un moment plus tardif."
                }
            
            current_pole = best_lap_times.min().total_seconds()
            
            try:
                best_s1 = current_data['Sector1Time'].dropna().min().total_seconds()
                best_s2 = current_data['Sector2Time'].dropna().min().total_seconds()
                best_s3 = current_data['Sector3Time'].dropna().min().total_seconds()
                ultimate_lap = best_s1 + best_s2 + best_s3
            except:
                ultimate_lap = current_pole * 0.98
            
        except:
            return None

        evo_gain = self.evolution_rate * time_remaining_min
        
        pred_low = ultimate_lap - (evo_gain * 0.5)
        pred_high = current_pole - evo_gain
        
        return {
            "success": True,
            "Cible": "P1 (Pole Position)",
            "Pole Provisoire": float(current_pole),
            "Ultimate Lap Théorique": float(ultimate_lap),
            "Intervalle Prédit": (float(pred_low), float(pred_high))
        }

    def get_session_info(self):
        """
        Retourne des informations sur la session.
        """
        if len(self.laps) == 0:
            return None
        
        min_time = self.laps['Time'].min()
        max_time = self.laps['Time'].max()
        
        # Compter les tours valides à différents moments
        valid_laps = self.laps[self.laps['LapTime'].notna()]
        
        return {
            "min_time": str(min_time),
            "max_time": str(max_time),
            "min_time_minutes": min_time.total_seconds() / 60 if min_time else 0,
            "max_time_minutes": max_time.total_seconds() / 60 if max_time else 0,
            "drivers": list(self.laps['Driver'].unique()),
            "total_valid_laps": len(valid_laps)
        }

# Cache global pour les prédicteurs
predictors_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/years')
def get_years():
    """Retourne les années disponibles pour la F1."""
    years = list(range(2018, 2025))
    return jsonify(years)

@app.route('/api/events/<int:year>')
def get_events(year):
    """Retourne les événements disponibles pour une année donnée."""
    try:
        schedule = fastf1.get_event_schedule(year)
        # Filtrer uniquement les événements qui ont une session de qualification
        events = []
        for idx, event in schedule.iterrows():
            # Vérifier si l'événement a des sessions (pas de test ou événement spécial)
            if pd.notna(event.get('EventName')) and pd.notna(event.get('RoundNumber')):
                events.append({
                    'round': int(event['RoundNumber']),
                    'name': event['EventName']
                })
        return jsonify(events)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/session-info/<int:year>/<path:event>')
def get_session_info(year, event):
    """Charge la session et retourne les informations."""
    try:
        cache_key = f"{year}_{event}"
        if cache_key not in predictors_cache:
            predictor = QualiPredictor(year, event, 'Q')
            predictors_cache[cache_key] = predictor
        else:
            predictor = predictors_cache[cache_key]
        
        info = predictor.get_session_info()
        if info:
            print(f"Session info: min={info['min_time_minutes']:.1f}min, max={info['max_time_minutes']:.1f}min, valid_laps={info['total_valid_laps']}")
            return jsonify(info)
        else:
            return jsonify({'error': 'Aucune donnée disponible'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """Effectue une prédiction."""
    try:
        data = request.json
        year = data.get('year')
        event = data.get('event')
        simulation_time = data.get('simulation_time')  # Minutes relatives (ex: 12 min pour Q1)
        time_remaining = data.get('time_remaining')
        session_type = data.get('session_type', 'Q1')  # Q1, Q2, Q3
        
        cache_key = f"{year}_{event}"
        if cache_key not in predictors_cache:
            predictor = QualiPredictor(year, event, 'Q')
            predictors_cache[cache_key] = predictor
        else:
            predictor = predictors_cache[cache_key]
        
        # Convertir le temps relatif en temps absolu selon la session
        session_range = predictor.session_ranges.get(session_type)
        if not session_range:
            return jsonify({'error': f'Plage de temps pour {session_type} non détectée'}), 400
        
        start_time, end_time = session_range
        session_duration = end_time - start_time
        
        # Calculer le temps absolu : début de session + temps relatif
        # Limiter au maximum de la session
        absolute_time = start_time + min(simulation_time, session_duration)
        
        print(f"DEBUG: Session {session_type}, Temps relatif: {simulation_time}min, Temps absolu: {absolute_time:.1f}min")
        
        if session_type in ['Q1', 'Q2']:
            result = predictor.predict_q1_cutoff(absolute_time, time_remaining)
        else:  # Q3
            result = predictor.predict_q3_pole(absolute_time, time_remaining)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Données insuffisantes pour faire une prédiction'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
