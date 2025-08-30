import { Injectable } from '@nestjs/common';
import { SendEmailDto } from './email.dto';

@Injectable()
export class EmailService {
  async sendEmails(sendEmailDto: SendEmailDto): Promise<any> {
    // Aquí implementarías la lógica para enviar los correos
    // Podrías usar nodemailer u otro paquete
    
    const results = [];
    const defaultSignature = sendEmailDto.signature || "Atentamente,\nEl equipo de proyecto";
    
    for (const recipient of sendEmailDto.recipients) {
      const emailContent = `${recipient.message}\n\n${defaultSignature}`;
      
      // Simulación de envío de correo
      results.push({
        recipient: recipient.name,
        status: 'success',
        content: emailContent,
        subject: sendEmailDto.subject
      });
      
      console.log(`Correo enviado a ${recipient.name} con asunto "${sendEmailDto.subject}"`);
    }
    
    return { success: true, results };
  }
}