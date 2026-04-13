import { Component, signal } from '@angular/core';

interface ChatMessage {
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  // Estado del chat usando Angular Signals para máxima reactividad
  messages = signal<ChatMessage[]>([
    { text: 'TERMINAL INICIADA. Esperando entrada del operador...', sender: 'bot', timestamp: new Date() }
  ]);
  
  inputText = signal('');

  // Actualiza el contenido del signal cuando el usuario escribe
  updateInput(event: Event) {
    const input = event.target as HTMLInputElement;
    this.inputText.set(input.value);
  }

  // Lógica de ejecución al enviar el mensaje
  sendMessage() {
    const query = this.inputText().trim();
    if (!query) return;

    // 1. Añadimos el mensaje del usuario al historial
    this.messages.update(msgs => [...msgs, {
      text: query,
      sender: 'user',
      timestamp: new Date()
    }]);

    // 2. Limpiamos la barra de entrada
    this.inputText.set('');

    // 3. (Mock temporal) Simulamos la latencia de red y proceso de la IA
    setTimeout(() => {
      this.messages.update(msgs => [...msgs, {
        text: 'EJECUTANDO ANÁLISIS SINTÁCTICO... [Simulación - API No Conectada]',
        sender: 'bot',
        timestamp: new Date()
      }]);
    }, 800); // 800ms de retraso para dar sensación de procesamiento
  }
}
