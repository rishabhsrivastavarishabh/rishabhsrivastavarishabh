# Flutter Integration for Next.js Backend

This document captures the Flutter-side integration scaffolding for a Next.js backend with Supabase authentication.

## Key Notes

- Use Flutter's `http` package for network requests.
- Expose Next.js API route handlers under `app/api/.../route.ts`.
- Keep `SUPABASE_SERVICE_ROLE_KEY` server-side only; never ship it to Flutter clients.
- On Vercel, environment variable changes require a redeploy before they affect production deployments.

## 1) `pubspec.yaml`

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.5.0
  shared_preferences: ^2.5.3
```

## 2) `lib/services/api_service.dart`

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiException implements Exception {
  final int statusCode;
  final String message;

  ApiException(this.statusCode, this.message);

  @override
  String toString() => 'ApiException($statusCode): $message';
}

class ApiService {
  // Update this after deployment
  static const String backendUrl = 'https://your-deployment.vercel.app';

  static Uri _uri(String path, [Map<String, dynamic>? query]) {
    return Uri.parse('$backendUrl$path').replace(
      queryParameters: query?.map((k, v) => MapEntry(k, v.toString())),
    );
  }

  static Map<String, String> _headers({String? accessToken}) {
    return {
      'Content-Type': 'application/json',
      if (accessToken != null && accessToken.isNotEmpty)
        'Authorization': 'Bearer $accessToken',
    };
  }

  static dynamic _decodeResponse(http.Response response) {
    final body = response.body.isEmpty ? '{}' : response.body;
    final decoded = jsonDecode(body);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      final message = decoded is Map && decoded['error'] != null
          ? decoded['error'].toString()
          : 'Request failed';
      throw ApiException(response.statusCode, message);
    }

    return decoded;
  }

  static Future<Map<String, dynamic>> createChat({
    required String accessToken,
    String title = 'New Chat',
    String mode = 'chat',
    String? projectId,
  }) async {
    final response = await http.post(
      _uri('/api/chats'),
      headers: _headers(accessToken: accessToken),
      body: jsonEncode({
        'title': title,
        'mode': mode,
        'project_id': projectId,
      }),
    );

    return Map<String, dynamic>.from(_decodeResponse(response));
  }

  static Future<List<dynamic>> getChats({
    required String accessToken,
  }) async {
    final response = await http.get(
      _uri('/api/chats'),
      headers: _headers(accessToken: accessToken),
    );

    final data = Map<String, dynamic>.from(_decodeResponse(response));
    return (data['chats'] as List?) ?? [];
  }

  static Future<Map<String, dynamic>> getChat({
    required String accessToken,
    required String chatId,
  }) async {
    final response = await http.get(
      _uri('/api/chats/$chatId'),
      headers: _headers(accessToken: accessToken),
    );

    return Map<String, dynamic>.from(_decodeResponse(response));
  }

  static Future<Map<String, dynamic>> updateChat({
    required String accessToken,
    required String chatId,
    String? title,
    String? mode,
    String? projectId,
  }) async {
    final response = await http.patch(
      _uri('/api/chats/$chatId'),
      headers: _headers(accessToken: accessToken),
      body: jsonEncode({
        if (title != null) 'title': title,
        if (mode != null) 'mode': mode,
        'project_id': projectId,
      }),
    );

    return Map<String, dynamic>.from(_decodeResponse(response));
  }

  static Future<void> deleteChat({
    required String accessToken,
    required String chatId,
  }) async {
    final response = await http.delete(
      _uri('/api/chats/$chatId'),
      headers: _headers(accessToken: accessToken),
    );

    if (response.statusCode != 204) {
      _decodeResponse(response);
    }
  }

  static Future<List<dynamic>> getMessages({
    required String accessToken,
    required String chatId,
  }) async {
    final response = await http.get(
      _uri('/api/chats/$chatId/messages'),
      headers: _headers(accessToken: accessToken),
    );

    final data = Map<String, dynamic>.from(_decodeResponse(response));
    return (data['messages'] as List?) ?? [];
  }

  static Future<Map<String, dynamic>> sendMessage({
    required String accessToken,
    required String chatId,
    required String role,
    required String content,
    Map<String, dynamic>? metadata,
  }) async {
    final response = await http.post(
      _uri('/api/chats/$chatId/messages'),
      headers: _headers(accessToken: accessToken),
      body: jsonEncode({
        'role': role,
        'content': content,
        'metadata': metadata ?? {},
      }),
    );

    return Map<String, dynamic>.from(_decodeResponse(response));
  }

  static Future<List<dynamic>> getProjects({
    required String accessToken,
  }) async {
    final response = await http.get(
      _uri('/api/projects'),
      headers: _headers(accessToken: accessToken),
    );

    final data = Map<String, dynamic>.from(_decodeResponse(response));
    return (data['projects'] as List?) ?? [];
  }

  static Future<Map<String, dynamic>> createProject({
    required String accessToken,
    required String title,
    String? description,
  }) async {
    final response = await http.post(
      _uri('/api/projects'),
      headers: _headers(accessToken: accessToken),
      body: jsonEncode({
        'title': title,
        'description': description,
      }),
    );

    return Map<String, dynamic>.from(_decodeResponse(response));
  }

  static Future<Map<String, dynamic>> updateProject({
    required String accessToken,
    required String projectId,
    String? title,
    String? description,
  }) async {
    final response = await http.patch(
      _uri('/api/projects/$projectId'),
      headers: _headers(accessToken: accessToken),
      body: jsonEncode({
        if (title != null) 'title': title,
        if (description != null) 'description': description,
      }),
    );

    return Map<String, dynamic>.from(_decodeResponse(response));
  }

  static Future<void> deleteProject({
    required String accessToken,
    required String projectId,
  }) async {
    final response = await http.delete(
      _uri('/api/projects/$projectId'),
      headers: _headers(accessToken: accessToken),
    );

    if (response.statusCode != 204) {
      _decodeResponse(response);
    }
  }

  static Future<Map<String, dynamic>> createShareLink({
    required String accessToken,
    required String chatId,
  }) async {
    final response = await http.post(
      _uri('/api/chats/$chatId/share'),
      headers: _headers(accessToken: accessToken),
    );

    return Map<String, dynamic>.from(_decodeResponse(response));
  }

  static Future<Map<String, dynamic>> getShareInfo({
    required String accessToken,
    required String chatId,
  }) async {
    final response = await http.get(
      _uri('/api/chats/$chatId/share'),
      headers: _headers(accessToken: accessToken),
    );

    return Map<String, dynamic>.from(_decodeResponse(response));
  }

  static Future<void> disableShareLink({
    required String accessToken,
    required String chatId,
  }) async {
    final response = await http.delete(
      _uri('/api/chats/$chatId/share'),
      headers: _headers(accessToken: accessToken),
    );

    if (response.statusCode != 204) {
      _decodeResponse(response);
    }
  }

  static Future<List<dynamic>> getSharedMessages({
    required String token,
  }) async {
    final response = await http.get(_uri('/api/share/$token'));

    final data = Map<String, dynamic>.from(_decodeResponse(response));
    return (data['messages'] as List?) ?? [];
  }
}
```

## 3) `lib/services/auth_service.dart`

```dart
import 'package:supabase_flutter/supabase_flutter.dart';

class AuthService {
  static final _client = Supabase.instance.client;

  static Future<void> signUp({
    required String email,
    required String password,
  }) async {
    await _client.auth.signUp(
      email: email,
      password: password,
    );
  }

  static Future<void> login({
    required String email,
    required String password,
  }) async {
    await _client.auth.signInWithPassword(
      email: email,
      password: password,
    );
  }

  static Future<void> logout() async {
    await _client.auth.signOut();
  }

  static String? get accessToken => _client.auth.currentSession?.accessToken;

  static User? get currentUser => _client.auth.currentUser;
}
```

## 4) `lib/main.dart`

```dart
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'screens/chat_screen.dart';
import 'screens/login_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Supabase.initialize(
    url: 'https://xhjfushlbshjyqljdwci.supabase.co',
    anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhoamZ1c2hsYnNoanlxbGpkd2NpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI0OTk1MDEsImV4cCI6MjA4ODA3NTUwMX0.iBaHJFe2ludfqC9bbeVlDNiUX3kxxNKIkTeId8O4SBQ',
  );

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final session = Supabase.instance.client.auth.currentSession;

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: session == null ? const LoginScreen() : const ChatScreen(),
    );
  }
}
```

## 5) `lib/screens/login_screen.dart`

```dart
import 'package:flutter/material.dart';
import '../services/auth_service.dart';
import 'chat_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  bool isLoading = false;
  bool isSignup = false;

  Future<void> submit() async {
    setState(() => isLoading = true);
    try {
      if (isSignup) {
        await AuthService.signUp(
          email: emailController.text.trim(),
          password: passwordController.text.trim(),
        );
      } else {
        await AuthService.login(
          email: emailController.text.trim(),
          password: passwordController.text.trim(),
        );
      }

      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const ChatScreen()),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString())),
      );
    } finally {
      if (mounted) setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(isSignup ? 'Sign Up' : 'Login')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: emailController,
              decoration: const InputDecoration(labelText: 'Email'),
            ),
            TextField(
              controller: passwordController,
              decoration: const InputDecoration(labelText: 'Password'),
              obscureText: true,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: isLoading ? null : submit,
              child: Text(isLoading
                  ? 'Please wait...'
                  : (isSignup ? 'Create Account' : 'Login')),
            ),
            TextButton(
              onPressed: () => setState(() => isSignup = !isSignup),
              child: Text(
                isSignup
                    ? 'Already have an account? Login'
                    : 'No account? Sign up',
              ),
            )
          ],
        ),
      ),
    );
  }
}
```

## 6) `lib/screens/chat_screen.dart`

```dart
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import 'login_screen.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final inputController = TextEditingController();
  final scrollController = ScrollController();

  List<dynamic> chats = [];
  List<dynamic> messages = [];
  String? activeChatId;
  bool isLoading = false;

  @override
  void initState() {
    super.initState();
    loadChats();
  }

  Future<void> loadChats() async {
    final token = AuthService.accessToken;
    if (token == null) return;

    setState(() => isLoading = true);
    try {
      final result = await ApiService.getChats(accessToken: token);
      setState(() => chats = result);

      if (result.isNotEmpty) {
        final firstChatId = result.first['chat_id'] as String;
        await openChat(firstChatId);
      }
    } catch (e) {
      showError(e);
    } finally {
      if (mounted) setState(() => isLoading = false);
    }
  }

  Future<void> createChat() async {
    final token = AuthService.accessToken;
    if (token == null) return;

    try {
      final result = await ApiService.createChat(accessToken: token);
      final chat = result['chat'] as Map<String, dynamic>;
      setState(() {
        chats.insert(0, chat);
        activeChatId = chat['chat_id'] as String;
        messages = [];
      });
    } catch (e) {
      showError(e);
    }
  }

  Future<void> openChat(String chatId) async {
    final token = AuthService.accessToken;
    if (token == null) return;

    try {
      final result = await ApiService.getMessages(
        accessToken: token,
        chatId: chatId,
      );

      setState(() {
        activeChatId = chatId;
        messages = result;
      });

      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (scrollController.hasClients) {
          scrollController.jumpTo(scrollController.position.maxScrollExtent);
        }
      });
    } catch (e) {
      showError(e);
    }
  }

  Future<void> sendMessage() async {
    final token = AuthService.accessToken;
    final text = inputController.text.trim();

    if (token == null || activeChatId == null || text.isEmpty) return;

    inputController.clear();

    try {
      final userMessage = await ApiService.sendMessage(
        accessToken: token,
        chatId: activeChatId!,
        role: 'user',
        content: text,
      );

      setState(() {
        messages.add(userMessage['message']);
      });

      final assistantMessage = await ApiService.sendMessage(
        accessToken: token,
        chatId: activeChatId!,
        role: 'assistant',
        content: 'Received: $text',
      );

      setState(() {
        messages.add(assistantMessage['message']);
      });

      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (scrollController.hasClients) {
          scrollController.animateTo(
            scrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 250),
            curve: Curves.easeOut,
          );
        }
      });
    } catch (e) {
      showError(e);
    }
  }

  Future<void> shareActiveChat() async {
    final token = AuthService.accessToken;
    if (token == null || activeChatId == null) return;

    try {
      final result = await ApiService.createShareLink(
        accessToken: token,
        chatId: activeChatId!,
      );

      final url = result['share_url']?.toString() ?? '';
      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Share link: $url')),
      );
    } catch (e) {
      showError(e);
    }
  }

  Future<void> logout() async {
    await AuthService.logout();

    if (!mounted) return;
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const LoginScreen()),
      (_) => false,
    );
  }

  void showError(Object e) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(e.toString())),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sarath'),
        actions: [
          IconButton(
            onPressed: createChat,
            icon: const Icon(Icons.add_comment_outlined),
          ),
          IconButton(
            onPressed: shareActiveChat,
            icon: const Icon(Icons.share),
          ),
          IconButton(
            onPressed: logout,
            icon: const Icon(Icons.logout),
          ),
        ],
      ),
      body: Row(
        children: [
          SizedBox(
            width: 260,
            child: Column(
              children: [
                const Padding(
                  padding: EdgeInsets.all(12),
                  child: Text(
                    'Chats',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ),
                Expanded(
                  child: ListView.builder(
                    itemCount: chats.length,
                    itemBuilder: (context, index) {
                      final chat = chats[index] as Map<String, dynamic>;
                      final chatId = chat['chat_id'] as String;
                      final title = (chat['title'] ?? 'Untitled').toString();

                      return ListTile(
                        selected: activeChatId == chatId,
                        title: Text(title),
                        subtitle: Text(chatId),
                        onTap: () => openChat(chatId),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
          const VerticalDivider(width: 1),
          Expanded(
            child: Column(
              children: [
                Expanded(
                  child: isLoading
                      ? const Center(child: CircularProgressIndicator())
                      : ListView.builder(
                          controller: scrollController,
                          itemCount: messages.length,
                          itemBuilder: (context, index) {
                            final msg = messages[index] as Map<String, dynamic>;
                            return ListTile(
                              title: Text(msg['role'].toString()),
                              subtitle: Text(msg['content'].toString()),
                            );
                          },
                        ),
                ),
                Padding(
                  padding: const EdgeInsets.all(12),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: inputController,
                          decoration: const InputDecoration(
                            hintText: 'Type a message...',
                            border: OutlineInputBorder(),
                          ),
                          onSubmitted: (_) => sendMessage(),
                        ),
                      ),
                      const SizedBox(width: 8),
                      ElevatedButton(
                        onPressed: sendMessage,
                        child: const Text('Send'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
```

## 7) Deploy notes

```bash
# Option A: Vercel
cd /home/user/sarath-backend
npx vercel --prod

# Option B: Local
npm run dev
```

Environment variables required in Vercel project configuration:

- `NEXT_PUBLIC_SUPABASE_URL=https://xhjfushlbshjyqljdwci.supabase.co`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhoamZ1c2hsYnNoanlxbGpkd2NpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI0OTk1MDEsImV4cCI6MjA4ODA3NTUwMX0.iBaHJFe2ludfqC9bbeVlDNiUX3kxxNKIkTeId8O4SBQ`
- `SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhoamZ1c2hsYnNoanlxbGpkd2NpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MjQ5OTUwMSwiZXhwIjoyMDg4MDc1NTAxfQ.lzh2x9G2-maFOhfaqu6pLCvlgKjfWNv_Jayob2qzU9g`

## 8) Security warning

Do **not** use the service role key in Flutter. Keep it backend-only for secure token resolution on share endpoints.
