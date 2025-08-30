import { Controller, Post, Body } from '@nestjs/common';
import { AudioService } from './audio.service';
import { ProcessAudioDto } from './audio.dto';

@Controller('audio')
export class AudioController {
  constructor(private readonly audioService: AudioService) {}

  @Post('process')
  async processAudio(@Body() processAudioDto: ProcessAudioDto) {
    return this.audioService.processAudio(processAudioDto.audioPath);
  }
}