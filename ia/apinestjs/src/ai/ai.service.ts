import { Injectable } from '@nestjs/common';
import axios from 'axios';
import { CreateTaskDto } from '../tasks/dto/create-task.dto';

@Injectable()
export class AiService {
  private readonly AI_API_URL = 'http://localhost:8000/prioritize';

  async predictPriority(taskData: CreateTaskDto): Promise<string> {
    try {
      const response = await axios.post(this.AI_API_URL, {
        title: taskData.title,
        dueDate: taskData.dueDate,
        importance: taskData.importance,
        duration: taskData.duration
      });
      return response.data.priority;
    } catch (error) {
      console.error('Error calling Python API:', error);
      return 'media';
    }
  }
}