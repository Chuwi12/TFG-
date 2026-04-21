import { Component, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';

interface ChatMessage {
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatResponse {
  intent: string;
  response: string;
  status: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/chat';

  messages = signal<ChatMessage[]>([
    { text: 'TERMINAL INICIADA. Esperando entrada del operador...', sender: 'bot', timestamp: new Date() }
  ]);

  inputText = signal('');

  updateInput(event: Event) {
    const input = event.target as HTMLInputElement;
    this.inputText.set(input.value);
  }

  sendMessage() {
    const query = this.inputText().trim();
    if (!query) return;

    // 1. Mensaje del usuario al historial
    this.messages.update(msgs => [...msgs, {
      text: query,
      sender: 'user',
      timestamp: new Date()
    }]);

    this.inputText.set('');

    // 2. Llamada REAL al backend de Python
    this.http.post<ChatResponse>(this.apiUrl, { message: query }).subscribe({
      next: (res) => {
        this.messages.update(msgs => [...msgs, {
          text: res.response,
          sender: 'bot',
          timestamp: new Date()
        }]);
      },
      error: () => {
        this.messages.update(msgs => [...msgs, {
          text: 'ERROR: No se pudo conectar con el servidor. ¿Está el backend activo?',
          sender: 'bot',
          timestamp: new Date()
        }]);
      }
    });
  }
}

