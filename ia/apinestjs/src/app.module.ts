import { Module } from '@nestjs/common';
import { AudioController } from './audio/audio.controller';
import { AudioService } from './audio/audio.service';
import { EmailController } from './email/email.controller';
import { EmailService } from './email/email.service';
import { PythonService } from './shared/python.service';

@Module({
  imports: [],
  controllers: [AudioController, EmailController],
  providers: [AudioService, EmailService, PythonService],
})
export class AppModule {}