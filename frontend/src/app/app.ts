import { Component, inject, signal, computed, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { UpperCasePipe } from '@angular/common';

interface ChatMessage {
  text: string;
  sender: 'user' | 'bot' | 'typing';
  timestamp: Date;
}

interface ChatResponse {
  response: string;
}

interface UserProfile {
  name: string;
  age: string;
  level: string;
  income: string;
  concern: string;
}

interface RoadmapProfile {
  title: string;
  concepts: string[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [UpperCasePipe],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements AfterViewChecked {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/chat';

  @ViewChild('scrollMe') private myScrollContainer!: ElementRef;

  // Onboarding
  onboardingComplete = signal(false);
  onboardingStep = signal(0);
  userProfile = signal<UserProfile>({ name: '', age: '', level: '', income: '', concern: '' });

  // Roadmap
  showRoadmap = signal(false);
  roadmapProfile = signal<RoadmapProfile>({ title: '', concepts: [] });

  // Chat — simple flat array, no sessions
  messages = signal<ChatMessage[]>([]);
  inputText = signal('');
  isLoading = signal(false);

  // Preguntas del onboarding
  onboardingQuestions = [
    { key: 'name', question: '¿Cómo te llamas?', placeholder: 'Tu nombre...' },
    { key: 'age', question: '¿Cuántos años tienes?', placeholder: 'Tu edad...' },
    { key: 'level', question: '¿Cuál es tu nivel de conocimiento financiero?', placeholder: '', options: ['Principiante', 'Intermedio', 'Avanzado'] },
    { key: 'income', question: '¿Tienes ingresos propios ahora mismo?', placeholder: '', options: ['Sí, trabajo', 'Sí, becas o ayudas', 'No de momento'] },
    { key: 'concern', question: '¿Qué te preocupa más?', placeholder: '', options: ['Empezar a ahorrar', 'Entender la inversión', 'Salir de deudas', 'No sé por dónde empezar'] },
  ];

  currentQuestion = computed(() => {
    const step = this.onboardingStep();
    return step < this.onboardingQuestions.length ? this.onboardingQuestions[step] : null;
  });

  onboardingInput = signal('');

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    try {
      const el = this.myScrollContainer?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    } catch (err) { }
  }

  // Onboarding
  updateOnboardingInput(event: Event) {
    const input = event.target as HTMLInputElement;
    this.onboardingInput.set(input.value);
  }

  selectOption(key: string, value: string) {
    this.userProfile.update(p => ({ ...p, [key]: value }));

    const step = this.onboardingStep();
    if (step + 1 < this.onboardingQuestions.length) {
      this.onboardingStep.set(step + 1);
    } else {
      this.finishOnboarding();
    }
  }

  submitOnboardingStep() {
    const value = this.onboardingInput().trim();
    if (!value) return;

    const step = this.onboardingStep();
    const key = this.onboardingQuestions[step].key;

    this.userProfile.update(p => ({ ...p, [key]: value }));
    this.onboardingInput.set('');

    if (step + 1 < this.onboardingQuestions.length) {
      this.onboardingStep.set(step + 1);
    } else {
      this.finishOnboarding();
    }
  }

  private generateProfile(): RoadmapProfile {
    const profile = this.userProfile();
    const concern = profile.concern;
    const income = profile.income;

    if (concern === 'Entender la inversión') {
      return {
        title: 'INVERSOR CURIOSO',
        concepts: ['Qué es un ETF', 'Fondos indexados', 'Riesgo y diversificación']
      };
    }

    if (concern === 'Salir de deudas') {
      return {
        title: 'GESTIÓN DE DEUDA',
        concepts: ['Método bola de nieve', 'Interés del crédito', 'Presupuesto de emergencia']
      };
    }

    // Default: "Empezar a ahorrar", "No sé por dónde empezar", or income "No de momento"
    return {
      title: 'AHORRADOR PRINCIPIANTE',
      concepts: ['Interés compuesto', 'Regla del 50/30/20', 'Cuenta de ahorro vs cuenta corriente']
    };
  }

  private finishOnboarding() {
    const rp = this.generateProfile();
    this.roadmapProfile.set(rp);
    this.showRoadmap.set(true);
  }

  startTerminal() {
    const profile = this.userProfile();
    const rp = this.roadmapProfile();
    const lines = [
      'Hola, ' + profile.name + '. Perfil registrado.',
      'Tipo: ' + rp.title + '.',
      'Ruta inicial:',
      '- ' + rp.concepts.join(String.fromCharCode(10) + '- '),
      '',
      'Estoy listo para ayudarte con tus finanzas personales.',
      'Pregunta con tus palabras: ahorro, deudas, inversion, presupuesto o seguridad.'
    ];
    const greeting = lines.join(String.fromCharCode(10));
    this.messages.set([{ text: greeting, sender: 'bot', timestamp: new Date() }]);
    this.onboardingComplete.set(true);
  }

  // Chat
  updateInput(event: Event) {
    const input = event.target as HTMLInputElement;
    this.inputText.set(input.value);
  }

  sendMessage() {
    const query = this.inputText().trim();
    if (!query || this.isLoading()) return;

    // 1. Mensaje del usuario al historial
    this.messages.update(msgs => [...msgs, {
      text: query,
      sender: 'user' as const,
      timestamp: new Date()
    }]);

    this.inputText.set('');
    this.isLoading.set(true);

    // 2. Indicador de carga
    this.messages.update(msgs => [...msgs, {
      text: '...',
      sender: 'typing' as const,
      timestamp: new Date()
    }]);

    // 3. Crear el "Prompt Oculto" con el contexto del usuario
    const contexto = `Actúa como asesor financiero. Soy un joven de nivel ${this.userProfile().level} y mi objetivo principal es ${this.userProfile().concern}. `;
    const promptOculto = contexto + query;

    // 4. Llamada REAL al backend de Python
    this.http.post<ChatResponse>(this.apiUrl, {
      message: promptOculto
    }).subscribe({
      next: (res) => {
        this.messages.update(msgs => [
          ...msgs.filter(m => m.sender !== 'typing'),
          { text: res.response, sender: 'bot' as const, timestamp: new Date() }
        ]);
        this.isLoading.set(false);
      },
      error: () => {
        this.messages.update(msgs => [
          ...msgs.filter(m => m.sender !== 'typing'),
          { text: 'ERROR: No se pudo conectar con el servidor. ¿Está el backend activo?', sender: 'bot' as const, timestamp: new Date() }
        ]);
        this.isLoading.set(false);
      }
    });
  }
}
