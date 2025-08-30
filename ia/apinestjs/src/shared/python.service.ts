import { Injectable } from '@nestjs/common';
import { spawn } from 'child_process';
import { join } from 'path';

@Injectable()
export class PythonService {
  private readonly pythonScriptPath = join(__dirname, '../../python-api/ejercicio9.py');

  async processAudio(audioPath: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', [this.pythonScriptPath, audioPath]);

      let result = '';
      let error = '';

      pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        error += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python process exited with code ${code}: ${error}`));
        } else {
          try {
            const parsedResult = JSON.parse(result);
            resolve(parsedResult);
          } catch (e) {
            resolve(result);
          }
        }
      });
    });
  }
}