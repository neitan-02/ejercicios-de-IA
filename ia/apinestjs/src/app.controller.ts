import { Controller, Post, Body } from '@nestjs/common';
import { AppService } from './app.service';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Post('process-audio')
  async processAudio(@Body() body: { audio: string }) {
    const audioBuffer = Buffer.from(body.audio, 'base64');
    return this.appService.processAudioInstructions(audioBuffer);
  }

  @Post('send-emails')
  async sendEmails(@Body() emailData: any) {
    return this.appService.sendEmails(emailData);
  }
}