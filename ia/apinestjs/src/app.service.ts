import { Injectable } from '@nestjs/common';
import { connect, NatsConnection } from 'nats'; // Importación actualizada

@Injectable()
export class AppService {
  private pythonServiceConnection: NatsConnection; // Cambiado de Client a NatsConnection

  constructor() {
    this.connectToPythonService();
  }

  private async connectToPythonService() {
    try {
      // Conexión actualizada a NATS
      this.pythonServiceConnection = await connect({ 
        servers: 'localhost:4222', // Puerto por defecto de NATS
        timeout: 5000 // Timeout de conexión
      });
      console.log('Conectado al servidor NATS');
    } catch (error) {
      console.error('Error conectando a NATS:', error);
      throw new Error('Failed to connect to NATS server');
    }
  }

  async processAudioInstructions(audioData: Buffer): Promise<any> {
    if (!this.pythonServiceConnection) {
      throw new Error('NATS connection not established');
    }

    try {
      // Enviar audio al servicio Python para procesamiento
      const response = await this.pythonServiceConnection.request(
        'process_audio', 
        audioData,
        { timeout: 10000 }
      );
      
      return JSON.parse(response.data.toString());
    } catch (error) {
      console.error('Error processing audio:', error);
      throw new Error('Failed to process audio instructions');
    }
  }

  async sendEmails(emailData: any): Promise<string> {
    const { recipients, subject, messages } = emailData;
    
    // Validación básica
    if (!recipients || !messages || recipients.length !== messages.length) {
      throw new Error('Invalid email data structure');
    }

    recipients.forEach((recipient: string, index: number) => {
      console.log(`Enviando correo a ${recipient} con asunto "${subject}"`);
      console.log(`Mensaje: ${messages[index]}`);
    });
    
    return 'Correos enviados exitosamente';
  }
}