import { Component, OnInit, inject, signal, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface ChatMessage {
  id: number;
  sender: 'bot' | 'user';
  text: string;
  type: 'text' | 'loading';
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit, AfterViewChecked {
  @ViewChild('chatScroll') private chatScrollContainer!: ElementRef;

  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000';

  messages = signal<ChatMessage[]>([]);
  private msgIdCounter = 0;

  userInput = '';
  isApiReady = signal<boolean>(true);

  ngOnInit() {
    this.addBotMessage('¡Hola! Soy tu asistente conversacional de IA basado en OpenAssistant (oasst1). Puedes hacerme cualquier pregunta o pedirme que genere texto en español.', 'text');
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    try {
      if (this.chatScrollContainer) {
        this.chatScrollContainer.nativeElement.scrollTop = this.chatScrollContainer.nativeElement.scrollHeight;
      }
    } catch(err) {}
  }

  private addBotMessage(text: string, type: 'text' | 'loading' = 'text') {
    this.messages.update(msgs => [...msgs, {
      id: this.msgIdCounter++,
      sender: 'bot',
      text,
      type
    }]);
  }

  private addUserMessage(text: string) {
    this.messages.update(msgs => [...msgs, {
      id: this.msgIdCounter++,
      sender: 'user',
      text,
      type: 'text'
    }]);
  }
  
  private removeLoadingMessage() {
    this.messages.update(msgs => msgs.filter(m => m.type !== 'loading'));
  }

  handleTextSubmit() {
    if (!this.userInput.trim()) return;
    const text = this.userInput.trim();
    this.userInput = '';
    
    this.addUserMessage(text);
    this.addBotMessage('Pensando...', 'loading');
    
    this.http.post<{response: string}>(`${this.apiUrl}/chat`, { message: text })
      .subscribe({
        next: (res) => {
          this.removeLoadingMessage();
          setTimeout(() => {
              this.addBotMessage(res.response || 'No logré generar una respuesta.');
          }, 100);
        },
        error: (err) => {
          console.error(err);
          this.removeLoadingMessage();
          this.addBotMessage('Hubo un error de conexión con la IA. Asegúrate de que el backend (main.py) esté encendido.');
        }
      });
  }

  formatText(text: string): string {
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
  }
}
