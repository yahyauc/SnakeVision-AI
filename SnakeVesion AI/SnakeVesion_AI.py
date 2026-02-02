"""
 SNAKE GAME - Contrôle par GESTES (version corrigée)
"""

import cv2
import numpy as np
import random
import time
import urllib.request
import os

# ============================================
# TÉLÉCHARGEMENT MODÈLE
# ============================================

def download_mediapipe_model():
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    model_path = "hand_landmarker.task"

    if not os.path.exists(model_path):
        print("Téléchargement modèle...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(" Modèle téléchargé")
        except:
            print("  Utilisation modèle par défaut")
            return None
    return model_path

# ============================================
# IMPORTATIONS
# ============================================

print("="*60)
print(" SNAKE GAME - CONTRÔLE PAR GESTES")
print("="*60)

try:
    import cv2, numpy as np, mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("Bibliotheques importees")
except ImportError as e:
    print(f"Erreur: {e}")
    print("Installez: pip install opencv-python mediapipe numpy")
    exit(1)

# ============================================
# CLASSES
# ============================================

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_tuple(self):
        return (self.x, self.y)
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class Direction:
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class GestureController:
    """Controleur par GESTES"""

    def __init__(self):
        print("  Initialisation controleur par gestes...")

        model_path = download_mediapipe_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.last_direction = None
        self.gesture_stability = 0
        self.min_stability = 3

        print("Contrôleur par gestes prêt!")

    def detect_hand_orientation(self, landmarks):
        """Détecte l'orientation de la main"""
        if len(landmarks) < 21:
            return None

        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]

        hand_vector_x = middle_tip.x - wrist.x
        hand_vector_y = middle_tip.y - wrist.y

        if abs(hand_vector_x) > abs(hand_vector_y):
            return Direction.RIGHT if hand_vector_x > 0 else Direction.LEFT
        else:
            return Direction.DOWN if hand_vector_y > 0 else Direction.UP

    def detect_open_hand(self, landmarks):
        """Détecte si la main est ouverte"""
        if len(landmarks) < 21:
            return False

        tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        mcps = [landmarks[2], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]

        extended = sum(1 for tip, mcp in zip(tips, mcps) if tip.y < mcp.y)
        return (extended >= 4)

    def process_frame(self, frame):
        """Traite une frame et détecte les gestes"""
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect(mp_image)

        direction = None
        is_open_hand = False
        hand_detected = False

        if result.hand_landmarks:
            hand_detected = True
            landmarks = result.hand_landmarks[0]

            current_orientation = self.detect_hand_orientation(landmarks)

            if current_orientation:
                if current_orientation == self.last_direction:
                    self.gesture_stability += 1
                else:
                    self.gesture_stability = 1
                    self.last_direction = current_orientation

                if self.gesture_stability >= self.min_stability:
                    direction = current_orientation
            else:
                self.last_direction = None
                self.gesture_stability = 0

            is_open_hand = self.detect_open_hand(landmarks)

            # Visualisation
            for idx, landmark in enumerate(landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # Squelette
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    start_x = int(start.x * w)
                    start_y = int(start.y * h)
                    end_x = int(end.x * w)
                    end_y = int(end.y * h)
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 200, 255), 2)

            if direction:
                dir_names = {
                    Direction.UP: "↑ HAUT",
                    Direction.DOWN: "↓ BAS",
                    Direction.LEFT: "← GAUCHE",
                    Direction.RIGHT: "→ DROITE"
                }
                dir_text = dir_names.get(direction, "")
                cv2.putText(frame, dir_text, (w//2 - 50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return direction, is_open_hand

    def release(self):
        pass

class SnakeGame:
    """Jeu Snake"""

    def __init__(self, speed=4):
        self.grid_size = 20
        self.cell_size = 30
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size

        self.speed = max(1, min(10, speed))
        self.update_interval = 0.8 * (1.0 / self.speed)

        self.colors = {
            'bg': (40, 40, 40),
            'snake_head': (0, 255, 0),
            'snake_body': (0, 180, 0),
            'food': (0, 0, 255),
            'score': (255, 255, 100)
        }

        self.reset()

    def reset(self):
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2

        self.snake = [
            Point(start_x, start_y),
            Point(start_x - 1, start_y),
            Point(start_x - 2, start_y)
        ]

        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.score = 0
        self.game_over = False
        self.food = self.generate_food()

    def generate_food(self):
        for _ in range(50):
            food = Point(
                random.randint(1, self.grid_size - 2),
                random.randint(1, self.grid_size - 2)
            )
            if food not in self.snake:
                return food
        return Point(self.grid_size - 3, self.grid_size - 3)

    def update(self):
        if self.game_over:
            return False

        current = self.direction
        next_dir = self.next_direction
        if (current[0] * -1, current[1] * -1) != next_dir:
            self.direction = next_dir

        head = self.snake[0]
        dx, dy = self.direction
        new_head = Point(head.x + dx, head.y + dy)

        if (new_head.x < 0 or new_head.x >= self.grid_size or
            new_head.y < 0 or new_head.y >= self.grid_size):
            self.game_over = True
            return False

        snake_tuples = [p.to_tuple() for p in self.snake]
        if new_head.to_tuple() in snake_tuples[1:]:
            self.game_over = True
            return False

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 10
            self.food = self.generate_food()
        else:
            self.snake.pop()

        return True

    def change_direction(self, direction):
        self.next_direction = direction

    def draw(self):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.colors['bg']

        for x in range(0, self.width, self.cell_size):
            cv2.line(canvas, (x, 0), (x, self.height), (50, 50, 50), 1)
        for y in range(0, self.height, self.cell_size):
            cv2.line(canvas, (0, y), (self.width, y), (50, 50, 50), 1)

        for i, segment in enumerate(self.snake):
            x1 = segment.x * self.cell_size
            y1 = segment.y * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size

            color = self.colors['snake_head'] if i == 0 else self.colors['snake_body']
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)

        fx1 = self.food.x * self.cell_size
        fy1 = self.food.y * self.cell_size
        cv2.rectangle(canvas,
                     (fx1 + 5, fy1 + 5),
                     (fx1 + self.cell_size - 5, fy1 + self.cell_size - 5),
                     self.colors['food'], -1)

        cv2.putText(canvas, f"SCORE: {self.score}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['score'], 2)

        if self.game_over:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
            canvas = cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0)

            texts = [
                ("GAME OVER", 1.5, (0, 0, 255), 3),
                (f"Score: {self.score}", 1.0, (255, 255, 100), 2),
                ("Main OUVERTE pour recommencer", 0.5, (200, 200, 200), 1)
            ]

            y_pos = self.height // 2 - 50
            for text, scale, color, thickness in texts:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
                text_x = (self.width - text_size[0]) // 2
                cv2.putText(canvas, text, (text_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
                y_pos += 50

        return canvas

class AISnakeGame:
    def __init__(self):
        self.cam_width = 640
        self.cam_height = 480
        self.game_width = 600
        self.game_height = 600

        print("\n" + "="*60)
        print("SNAKE GAME - CONTRÔLE PAR GESTES")
        print("="*60)

        self.game = SnakeGame(speed=4)
        self.gesture_controller = GestureController()

        self.running = True
        self.last_update = time.time()

        print("\n MODE GESTES ACTIVÉ")
        print("Orientez votre MAIN pour diriger")
        print("Main OUVERTE = Recommencer")
        print("\n Préparez votre main...")
        time.sleep(2)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(" Caméra non disponible")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        print("\n JEU LANCÉ! Montrez votre main...")

        last_fps_time = time.time()
        frame_count = 0
        fps = 0  # ← INITIALISATION ICI, c'est la correction clé!

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            current_time = time.time()

            # Détection gestes
            direction, is_open_hand = self.gesture_controller.process_frame(frame)

            # Gestion gestes
            if is_open_hand and self.game.game_over:
                self.game.reset()
                print(" Redémarré (main ouverte)")
                time.sleep(0.5)

            if direction and not self.game.game_over:
                self.game.change_direction(direction)

            # Mise à jour jeu
            if current_time - self.last_update >= self.game.update_interval:
                if not self.game.game_over:
                    self.game.update()
                self.last_update = current_time

            # Calcul FPS (CORRIGÉ)
            frame_count += 1
            if current_time - last_fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_time = current_time

            # Dessin jeu
            game_canvas = self.game.draw()

            # Combiner vues
            cam_resized = cv2.resize(frame, (self.cam_width, self.game_height))
            combined = np.zeros((self.game_height, self.cam_width + self.game_width, 3), dtype=np.uint8)
            combined[:, :self.cam_width] = cam_resized
            combined[:, self.cam_width:] = cv2.resize(game_canvas, (self.game_width, self.game_height))

            # Séparateur
            cv2.line(combined, (self.cam_width, 0),
                    (self.cam_width, self.game_height), (255, 255, 255), 2)

            # Titres
            cv2.putText(combined, "VOS GESTES", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "SNAKE GAME", (self.cam_width + 20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2)

            # FPS (MAINTENANT TOUJOURS DÉFINI)
            cv2.putText(combined, f"FPS: {fps}", (self.cam_width + 20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Score
            cv2.putText(combined, f"SCORE: {self.game.score}", (self.cam_width + 20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)

            # Afficher
            cv2.imshow(" Snake Game - Contrôle par Gestes", combined)

            # Touches clavier
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                self.game.reset()
                print(" Redémarré (touche R)")
            elif key == ord('+') and self.game.speed < 10:
                self.game.speed += 1
                self.game.update_interval = 0.8 * (1.0 / self.game.speed)
                print(f" Vitesse: {self.game.speed}/10")
            elif key == ord('-') and self.game.speed > 1:
                self.game.speed -= 1
                self.game.update_interval = 0.8 * (1.0 / self.game.speed)
                print(f" Vitesse: {self.game.speed}/10")

        # Nettoyage
        cap.release()
        self.gesture_controller.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print(f" Score final: {self.game.score}")
        print(" Merci d'avoir joué!")
        print("="*60)

# ============================================
# LANCEMENT
# ============================================

def main():
    print("="*60)
    print(" SNAKE GAME - CONTRÔLE PAR ORIENTATION DE MAIN")
    print("="*60)

    try:
        game = AISnakeGame()
        game.run()
    except KeyboardInterrupt:
        print("\n  Interrompu")
    except Exception as e:
        print(f"\n Erreur: {e}")
        import traceback
        traceback.print_exc()

    input("\nAppuyez sur Entrée...")

if __name__ == "__main__":
    main()
