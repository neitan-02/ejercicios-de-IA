import { Injectable } from '@nestjs/common';
import { PythonService } from '../shared/python.service';

@Injectable()
export class AudioService {
  constructor(private readonly pythonService: PythonService) {}

  async processAudio(audioPath: string): Promise<any> {
    return this.pythonService.processAudio(audioPath);
  }
}