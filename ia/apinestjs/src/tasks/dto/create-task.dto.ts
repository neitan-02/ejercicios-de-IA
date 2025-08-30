import { IsString, IsDateString, IsInt, Min, Max, IsNumber } from 'class-validator';

export class CreateTaskDto {
  @IsString()
  title: string;

  @IsDateString()
  dueDate: string;

  @IsInt()
  @Min(1)
  @Max(5)
  importance: number;

  @IsNumber()
  @Min(0.5)
  duration: number;
}