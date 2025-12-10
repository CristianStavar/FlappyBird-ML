import socket
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os


class FlappyBirdServer:
    def __init__(self):
        print("Iniciando servidor")

        self.host = '127.0.0.1'
        self.port = 9090
        self.running = True



        self.input_size = 5
        self.hidden_size = 24
        self.output_size = 2
        self.batch_size = 32
        self.memory_size = 10000
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001

        os.makedirs('models', exist_ok=True)
        self.setup_model()


        self.memory = deque(maxlen=self.memory_size)
        self.scores = []
        self.episodes = 0
        self.max_score = 0

        self.current_state = None
        self.current_action = None
        self.current_score = 0



    def setup_model(self):
        try:
            self.model = DQN(self.input_size, self.hidden_size, self.output_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()
            print("✅ Modelo de red neuronal creado")
        except Exception as e:
            print(f"❌ Error creando el modelo:  {e} ")
            raise

    def start_server(self):
        try:

            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(30.0)

            print(f"✅ Servidor escuchando en {self.host}: {self.port}")
            print(" Esperando conexión de Godot")

            self.accept_connections()

        except Exception as e:
            print(f"❌ Error iniciando servidor: {e}")



    def accept_connections(self):
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                print(f"✅ Conexión establecida desde: {addr}")
                self.handle_client(conn)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Error aceptando conexión: {e}")






    def handle_client(self, conn):
        try:
            conn.settimeout(0.1)

            while self.running:
                try:
                    data = conn.recv(4096)
                    #print(f"Este es data:  {data}")
                    if not data:
                        break


                    message = data.decode('utf-8').strip()
                    if message:
                        response = self.process_message(message)
                        if response:
                            #print(f"Esta es la respuesta :  {response}")
                            conn.sendall(response.encode('utf-8'))

                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"❌ Error con cliente: {e}")
                    break

        except Exception as e:
            print(f"❌ Error manejando cliente: {e}")
        finally:
            conn.close()
            print(" ####  Cliente desconectado  ####")


    def process_message(self, message):
        try:
            data = json.loads(message)
            print(f"Este es data:  {data}")
            if data.get("type") == "state":
                #print("  Es STATE")
                state = np.array(data["state"], dtype=np.float32)
                reward = data.get("reward", 0)
                self.current_score+=reward
                print(f"  \n       Este es reward:   {reward} ")
                done = data.get("done", False)


                if self.current_state is not None:
                    self.remember(self.current_state, self.current_action, reward, state, done)
                    self.replay()


                action = self.choose_action(state)
                self.current_state = state
                self.current_action = action


                if done:
                    self.episode_end()
                jason={"action":int(action)}
                print(f"Data procesado : {self.current_action}")
                return json.dumps(jason)


        except json.JSONDecodeError as e:
            print(f"❌ Error decodificando JSON: {e}")



    def choose_action(self, state):
        #print(" 🔎🔎   Eligiendo accion")
        if np.random.rand() <= self.epsilon:
            #print("Entra en IF")
            return random.randint(0, self.output_size - 1)
        else:
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()



    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))



    def replay(self):

        print("+++++++++ Estoy replayando")
        if len(self.memory) < self.batch_size:
            print("----❌------------❌-------- Dejo de replayar sin entrar")
            return

        self.model.train()
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)


        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)


        current_q = self.model(states)[range(self.batch_size), actions]

        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
            #print(f"f QQQQQQQQQQQQQQQ target {target_q}")



        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()






    def episode_end(self):
        self.episodes += 1
        self.scores.append(self.current_score)

        if self.current_score > self.max_score:
            self.max_score = self.current_score
            self.save_model("models/best_model.pth")

        print(
            f"✅✅ Episodio {self.episodes} - Score: {self.current_score} - Mejor: {self.max_score} - ε: {self.epsilon:.3f}")


        if self.episodes % 10 == 0:
            self.save_model(f"models/checkpoint_{self.episodes:04d}.pth")
            self.save_model("models/last_checkpoint.pth")


        self.current_state = None
        self.current_action = None
        self.current_score = 0
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):

        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'scores': self.scores,
                'episodes': self.episodes,
                'max_score': self.max_score,
                'memory':self.memory
            }, filename)
            print(f"✅ Modelo guardado: {filename}")
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")


    def load_model(self, filename):

        try:
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.scores = checkpoint['scores']
            self.episodes = checkpoint['episodes']
            self.max_score = checkpoint['max_score']


            if 'memory' in checkpoint:
                self.memory = checkpoint['memory']
            print(f"✅ Modelo cargado: {filename}  - con epsilon::: {self.epsilon}")

        except FileNotFoundError:
            print("⚠️  No se encontró modelo previo, empezando desde cero")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


def main():

    print("    FLAPPY BIRD AI - SERVIDOR DE ENTRENAMIENTO\n")


    server = FlappyBirdServer()


    server.load_model("models/last_checkpoint.pth")


    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n❌❌ Servidor interrumpido por el usuario ❌❌")
    except Exception as e:
        print(f"\n❌❌❌❌❌❌ Error crítico: {e}")
    finally:
        server.running = False
        server.save_model("models/final_model.pth")
        print("✅ Modelo final guardado")
        print(" Servidor cerrado")


if __name__ == "__main__":
    main()