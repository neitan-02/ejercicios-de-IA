import { Injectable } from '@nestjs/common';
import { Model } from 'mongoose';
import { InjectModel } from '@nestjs/mongoose';
import { CreateTaskDto } from './dto/create-task.dto';
import { Task } from './schemas/task.schema';
import { AiService } from '../ai/ai.service';

@Injectable()
export class TasksService {
  constructor(
    @InjectModel(Task.name) private taskModel: Model<Task>,
    private readonly aiService: AiService,
  ) {}

  async create(createTaskDto: CreateTaskDto): Promise<Task> {
    const priority = await this.aiService.predictPriority(createTaskDto);
    const createdTask = new this.taskModel({
      ...createTaskDto,
      dueDate: new Date(createTaskDto.dueDate),
      priority,
    });
    return createdTask.save();
  }

  async findAll(): Promise<Task[]> {
    return this.taskModel.find().sort({ 
      priority: 1,
      dueDate: 1,
      importance: -1
    }).exec();
  }
}