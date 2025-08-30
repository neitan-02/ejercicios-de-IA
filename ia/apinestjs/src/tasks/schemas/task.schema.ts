import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

@Schema()
export class Task extends Document {
  @Prop({ required: true })
  title: string;

  @Prop({ required: true, type: Date })
  dueDate: Date;

  @Prop({ required: true, min: 1, max: 5 })
  importance: number;

  @Prop({ required: true })
  duration: number;

  @Prop({ enum: ['alta', 'media', 'baja'] })
  priority: string;
}

export const TaskSchema = SchemaFactory.createForClass(Task);