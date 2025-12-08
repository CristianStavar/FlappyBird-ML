import socket
import time


class TestServer:
    def __init__(self, port=9090):
        self.port = port

    def start(self):
        print(" Iniciando servidor")

        while True:
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind(('127.0.0.1', self.port))
                server_socket.listen(1)
                server_socket.settimeout(30.0)

                print(f"✅ Servidor escuchando en 127.0.0.1: {self.port}")
                print(" Esperando conexión de Godot")

                client_socket, addr = server_socket.accept()
                print(f" CONEXIÓN ACEPTADA desde:  {addr}")
                client_socket.settimeout(.2)

                print("✅✅✅ Conexión establecida ✅✅✅")


                while True:
                    try:
                        data = client_socket.recv(1024)
                        if data:
                            print(f" Datos recibidos ({len(data)} bytes)")
                            print(f" Contenido: {data.decode('utf-8')}")


                            response = b'{"action": 1}'
                            client_socket.send(response)
                            print(" ----- Respuesta enviada ")
                        else:
                            print(" Cliente cerró")
                            break

                    except socket.timeout:

                        continue
                    except Exception as e:
                        print(f"❌ Error en comunicación: {e}")
                        break

            except socket.timeout:
                print(" Timeout esperando conexión - Reiniciando")
                continue
            except Exception as e:
                print(f" Error del servidor: {e}")
                print(" Reiniciando en 5 segundos")
                time.sleep(5)
            finally:
                try:
                    server_socket.close()
                except:
                    pass


if __name__ == "__main__":
    server = TestServer()
    server.start()
