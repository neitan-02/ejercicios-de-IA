import { Injectable, NotFoundException } from '@nestjs/common';
import { User } from './schemas/user.schemas';
import { Model } from 'mongoose';
import { InjectModel } from '@nestjs/mongoose';
import { CreateUserDto } from './dto/create-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';
import * as bcrypt from 'bcrypt';

@Injectable()
export class UsersService {
    constructor(
        @InjectModel(User.name) private userModel: Model<User>
    ) {}

    async login(user: UpdateUserDto) {
        try {
            const register = await this.userModel.findOne({ username: user.username }).exec();
            
            if (!register) {
                return false;
            }

            const isPasswordValid = await bcrypt.compare(user.password, register.password);
            return isPasswordValid ? register : false;
        } catch (error) {
            return false;
        }
    }

    async create(user: CreateUserDto) {
        const saltOrRounds = 10;
        const hash = await bcrypt.hash(user.password, saltOrRounds);
        const register = { ...user, password: hash };
        const createdUser = new this.userModel(register);
        return createdUser.save();
    }

    async findAll() {
        return this.userModel.find().exec();
    }

    async findOne(id: string) {
        const user = await this.userModel.findById(id).exec();
        if (!user) {
            throw new NotFoundException('User not found');
        }
        return user;
    }

    async update(id: string, user: UpdateUserDto) {
        if (user.password) {
            const saltOrRounds = 10;
            const hash = await bcrypt.hash(user.password, saltOrRounds);
            user.password = hash;
        }

        const updatedUser = await this.userModel.findByIdAndUpdate(
            id,
            user,
            { new: true }
        ).exec();

        if (!updatedUser) {
            throw new NotFoundException('User not found');
        }

        return updatedUser;
    }

    async remove(id: string) {
        const deletedUser = await this.userModel.findByIdAndDelete(id).exec();
        if (!deletedUser) {
            throw new NotFoundException('User not found');
        }
        return deletedUser;
    }
}