/**
 * 协议边界回归测试
 * ------------------------------------------------------------
 * 验证 ST-Memory-Context 在发往 OpenAI Chat Completions 兼容端点
 * （如 OpenCode Go）前，不会把插件私有字段泄漏进请求体。
 *
 * 被测逻辑与 index.js 中的实现保持一致：
 *   INTERNAL_MESSAGE_FIELDS / stripInternalMessageFields / sanitizeOutgoingRequestBody
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

// ===== 与 index.js 中实现保持一致的被测代码 =====
const INTERNAL_MESSAGE_FIELDS = Object.freeze([
    'isGaigaiData',
    'isGaigaiPrompt',
    'isGaigaiVector',
    'isPhoneMessage',
    '_toDelete'
]);

const stripInternalMessageFields = (node, seen = new WeakSet()) => {
    if (!node || typeof node !== 'object') return 0;
    if (seen.has(node)) return 0;
    seen.add(node);

    let removed = 0;
    if (Array.isArray(node)) {
        for (const item of node) {
            removed += stripInternalMessageFields(item, seen);
        }
        return removed;
    }

    const isMessageLike = typeof node.role === 'string';
    if (isMessageLike) {
        for (const key of INTERNAL_MESSAGE_FIELDS) {
            if (Object.prototype.hasOwnProperty.call(node, key)) {
                delete node[key];
                removed += 1;
            }
        }
    }
    // 无论当前节点是否为消息，都要继续向下递归
    for (const key of Object.keys(node)) {
        const value = node[key];
        if (value && typeof value === 'object') {
            removed += stripInternalMessageFields(value, seen);
        }
    }
    return removed;
};

const sanitizeOutgoingRequestBody = (requestArgs) => {
    const init = requestArgs && requestArgs[1];
    if (!init || typeof init.body !== 'string' || !init.body) return;
    try {
        const parsed = JSON.parse(init.body);
        const removed = stripInternalMessageFields(parsed);
        if (removed > 0) {
            init.body = JSON.stringify(parsed);
        }
    } catch {
        // 非 JSON 请求体原样放行
    }
};

// ===== 辅助工具 =====
const makeRequest = (bodyObject) => [
    '/api/backends/chat-completions/generate',
    { method: 'POST', body: JSON.stringify(bodyObject) }
];

/** 断言 messages 中不存在任何插件私有字段（按对象键判断，而非文本匹配） */
const assertNoInternalFields = (messages) => {
    const walk = (node, path) => {
        if (!node || typeof node !== 'object') return;
        if (Array.isArray(node)) {
            node.forEach((item, i) => walk(item, `${path}[${i}]`));
            return;
        }
        for (const field of INTERNAL_MESSAGE_FIELDS) {
            assert.ok(
                !Object.prototype.hasOwnProperty.call(node, field),
                `${path} 不应含有私有字段 ${field}`
            );
        }
        // 递归检查嵌套对象（tool_calls / content 等）
        for (const key of Object.keys(node)) {
            if (node[key] && typeof node[key] === 'object') {
                walk(node[key], `${path}.${key}`);
            }
        }
    };
    walk(messages, 'messages');
};

/** 断言消息只包含协议允许的字段 */
const ALLOWED_FIELDS = new Set([
    'role', 'content', 'name', 'tool_calls', 'tool_call_id', 'tool_choice',
    'function_call', 'function', 'reasoning', 'reasoning_content',
    'refusal', 'audio', 'parts', 'id', 'model', '_error'
]);

const assertStandardFieldsOnly = (message, label) => {
    for (const key of Object.keys(message)) {
        assert.ok(
            ALLOWED_FIELDS.has(key),
            `${label}: 出现非协议字段 "${key}"`
        );
    }
};

// ===== 测试用例 =====

test('场景1: 新建空白聊天 —— 空消息列表正常发出', () => {
    const args = makeRequest({
        messages: [{ role: 'system', content: 'You are a helpful assistant.' }],
        model: 'gpt-test',
        stream: true
    });

    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assert.equal(body.messages.length, 1);
    assert.equal(body.messages[0].role, 'system');
    assert.equal(body.model, 'gpt-test');
    assert.equal(body.stream, true);
});

test('场景2: 普通多轮对话 —— 完整携带历史，且不被破坏', () => {
    const history = [
        { role: 'system', content: 'System prompt' },
        { role: 'user', content: '你好' },
        { role: 'assistant', content: '你好，有什么可以帮你的？' },
        { role: 'user', content: '今天天气如何' }
    ];

    const args = makeRequest({ messages: structuredClone(history), model: 'm' });
    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assert.deepEqual(body.messages, history, '历史消息应被完整保留');
});

test('场景3: 含 isGaigaiData 的历史消息 —— 仅剥离私有字段，保留内容', () => {
    const args = makeRequest({
        messages: [
            {
                role: 'system',
                name: 'SYSTEM (记忆)',
                content: '【记忆只读数据库】历史存档内容',
                isGaigaiData: true        // 这就是触发 400 的字段
            },
            { role: 'user', content: '继续剧情' }
        ],
        model: 'm'
    });

    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assertNoInternalFields(body.messages);

    // 内容必须保留（不能因为清洗而丢记忆）
    assert.equal(body.messages[0].role, 'system');
    assert.equal(body.messages[0].name, 'SYSTEM (记忆)');
    assert.equal(body.messages[0].content, '【记忆只读数据库】历史存档内容');
    assertStandardFieldsOnly(body.messages[0], 'messages[0]');

    // 业务消息不受影响
    assert.deepEqual(body.messages[1], { role: 'user', content: '继续剧情' });
});

test('场景4: 带供应商状态的 Assistant 消息 —— 保留合法的 reasoning_tool 状态', () => {
    const args = makeRequest({
        messages: [
            { role: 'user', content: '算一下 1+1' },
            {
                role: 'assistant',
                content: '',
                reasoning_content: '需要考虑加法',
                tool_calls: [{
                    id: 'call_1',
                    type: 'function',
                    function: { name: 'calc', arguments: '{"a":1,"b":1}' }
                }],
                isGaigaiData: true
            },
            { role: 'tool', tool_call_id: 'call_1', content: '2' }
        ],
        model: 'm'
    });

    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assertNoInternalFields(body.messages);

    const assistant = body.messages[1];
    // 供应商相关的合法状态必须保留，不能伪造也不能丢弃
    assert.equal(assistant.reasoning_content, '需要考虑加法');
    assert.equal(assistant.tool_calls.length, 1);
    assert.equal(assistant.tool_calls[0].function.name, 'calc');
    // 不伪造 provider_metadata
    assert.equal(assistant.provider_metadata, undefined);
});

test('场景5: Tool Calls 历史 —— 结构完整（含嵌套字段清理）', () => {
    const args = makeRequest({
        messages: [
            { role: 'user', content: '查天气' },
            {
                role: 'assistant',
                content: null,
                tool_calls: [{
                    id: 'call_abc',
                    type: 'function',
                    function: { name: 'get_weather', arguments: '{"city":"北京"}' }
                }]
            },
            { role: 'tool', tool_call_id: 'call_abc', content: '{"temp":25}' }
        ],
        tools: [{
            type: 'function',
            function: { name: 'get_weather', parameters: { type: 'object', properties: {} } }
        }],
        model: 'm'
    });

    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assert.equal(body.messages[1].tool_calls[0].id, 'call_abc');
    assert.equal(body.messages[1].tool_calls[0].type, 'function');
    assert.equal(body.messages[1].tool_calls[0].function.arguments, '{"city":"北京"}');
    assert.equal(body.messages[2].tool_call_id, 'call_abc');
    assert.equal(body.tools[0].function.name, 'get_weather');
    assertNoInternalFields(body.messages);
});

test('场景6: 多模态消息 —— 保留原始 content 数组结构', () => {
    const multimodal = {
        role: 'user',
        content: [
            { type: 'text', text: '这张图是什么' },
            { type: 'image_url', image_url: { url: 'https://example.com/a.png', detail: 'high' } }
        ]
    };
    const args = makeRequest({ messages: [multimodal], model: 'm' });

    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assert.deepEqual(body.messages[0], multimodal, '多模态结构必须原样保留');
    assert.equal(body.messages[0].content.length, 2);
    assert.equal(body.messages[0].content[1].type, 'image_url');
});

test('场景7: 历史保存后重新加载 —— 反序列化后再次清洗依然安全', () => {
    // 模拟从存档读回：字段已被持久化
    const persisted = JSON.stringify([
        { role: 'system', content: '记忆表格', isGaigaiData: true, name: 'SYSTEM (表1)' },
        { role: 'user', content: '第一句' },
        { role: 'assistant', content: '第一答', isGaigaiPrompt: true }
    ]);
    const reloaded = JSON.parse(persisted);

    const args = makeRequest({ messages: reloaded, model: 'm' });
    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assertNoInternalFields(body.messages);
    assert.equal(body.messages.length, 3, '楼层数量不能变');
    assert.equal(body.messages[2].content, '第一答');

    // 幂等性：再次清洗不应报错或改变结果
    const firstPass = args[1].body;
    sanitizeOutgoingRequestBody(args);
    assert.equal(args[1].body, firstPass, '清洗必须幂等');
});

test('场景8: 所有私有字段变体都能被清理', () => {
    const args = makeRequest({
        messages: [
            { role: 'system', content: 'a', isGaigaiData: true },
            { role: 'system', content: 'b', isGaigaiPrompt: true },
            { role: 'user', content: 'c', isPhoneMessage: true },
            { role: 'assistant', content: 'd', isGaigaiVector: true },
            { role: 'user', content: 'e', _toDelete: true }
        ],
        model: 'm'
    });

    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assertNoInternalFields(body.messages);
    assert.equal(body.messages.length, 5, '不得删除楼层');
    body.messages.forEach((m, i) => assertStandardFieldsOnly(m, `messages[${i}]`));
});

test('场景9: 兼容性 —— 非 JSON 请求体与无 body 请求不受影响', () => {
    const binaryArgs = ['/api/images/generate', { method: 'POST', body: 'not-json-data' }];
    assert.doesNotThrow(() => sanitizeOutgoingRequestBody(binaryArgs));
    assert.equal(binaryArgs[1].body, 'not-json-data', '非 JSON 请求体应原样放行');

    assert.doesNotThrow(() => sanitizeOutgoingRequestBody(['/url', { method: 'GET' }]));
    assert.doesNotThrow(() => sanitizeOutgoingRequestBody(null));
});

test('场景10: 不误伤业务数据 —— 非消息节点的同名字段不被删除', () => {
    const args = makeRequest({
        messages: [{ role: 'user', content: 'x' }],
        // 顶层业务字段，不含 role，不应被当作消息处理
        metadata: { isGaigaiData: 'keep-me' },
        model: 'm'
    });

    sanitizeOutgoingRequestBody(args);

    const body = JSON.parse(args[1].body);
    assert.equal(body.metadata.isGaigaiData, 'keep-me', '非消息节点不应被清洗');
});

test('场景11: 最终 HTTP JSON 无任何不受支持的内部扩展字段', () => {
    const args = makeRequest({
        messages: [
            { role: 'system', content: 'P', isGaigaiData: true, isGaigaiVector: true },
            { role: 'user', content: 'U', isPhoneMessage: true, _toDelete: true },
            { role: 'assistant', content: 'A', isGaigaiPrompt: true },
            { role: 'user', content: 'U2' }
        ],
        model: 'opencode-go-model',
        temperature: 0.8,
        stream: true
    });

    sanitizeOutgoingRequestBody(args);

    const raw = args[1].body;
    // 这是"最终发出的字符串"层面的断言：检查 JSON 键，而非正文文本
    for (const field of INTERNAL_MESSAGE_FIELDS) {
        assert.ok(
            !raw.includes(`"${field}":`),
            `最终请求字符串仍含字段 ${field}`
        );
    }

    const body = JSON.parse(raw);
    assert.equal(body.messages.length, 4);
    assert.equal(body.temperature, 0.8);
    assert.equal(body.stream, true);
    body.messages.forEach((m, i) => assertStandardFieldsOnly(m, `messages[${i}]`));

    // 确认没有伪造 provider_metadata
    assert.ok(!raw.includes('provider_metadata'), '不得伪造 provider_metadata');
});
