export class RecipientDto {
  readonly name: string;
  readonly message: string;
}

export class SendEmailDto {
  readonly recipients: RecipientDto[];
  readonly subject: string;
  readonly signature?: string;
}